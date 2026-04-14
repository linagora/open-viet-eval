import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s  %(levelname)-8s  %(message)s",
	datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_block(text: str) -> dict[str, Any]:
	"""Extract the first JSON object from model output."""
	if not text:
		return {}
	match = JSON_BLOCK_PATTERN.search(text)
	if not match:
		return {}
	try:
		return json.loads(match.group(0))
	except json.JSONDecodeError:
		return {}


def clamp_score(value: Any, low: int = 1, high: int = 5) -> int:
	"""Convert a model score into a bounded integer score."""
	try:
		score = int(round(float(value)))
	except (TypeError, ValueError):
		return low
	return max(low, min(high, score))


async def call_judge(
	client: AsyncOpenAI,
	semaphore: asyncio.Semaphore,
	model: str,
	system_prompt: str,
	user_prompt: str,
	max_tokens: int,
	temperature: float,
) -> dict[str, Any]:
	"""Call judge model and parse strict JSON response."""
	async with semaphore:
		response = await client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			max_tokens=max_tokens,
			temperature=temperature,
		)
	content = (response.choices[0].message.content or "").strip()
	payload = parse_json_block(content)
	payload["_raw_response"] = content
	return payload


def read_text_if_exists(path: Path) -> str:
	if not path.exists():
		return ""
	return path.read_text(encoding="utf-8", errors="ignore").strip()


def find_vims_gold_summary(vims_summary_root: Path, cluster_name: str) -> str:
	"""
	Find first annotator summary for a cluster.
	Primary expected location: <vims_summary_root>/<cluster>/0.gold.txt
	Also supports: <vims_summary_root>/<cluster>/summary/0.gold.txt
	"""
	cluster_dir = vims_summary_root / cluster_name

	for expected in (
		cluster_dir / "0.gold.txt",
		cluster_dir / "summary" / "0.gold.txt",
	):
		text = read_text_if_exists(expected)
		if text:
			return text

	for search_dir in (cluster_dir, cluster_dir / "summary"):
		if search_dir.exists():
			for candidate in sorted(search_dir.glob("*0.gold*.txt")):
				text = read_text_if_exists(candidate)
				if text:
					return text
			for candidate in sorted(search_dir.glob("*.txt")):
				text = read_text_if_exists(candidate)
				if text:
					return text

	return ""


def summarize_criteria(df: pd.DataFrame, criteria: list[str]) -> dict[str, float]:
	if df.empty:
		return {f"avg_{c}": 0.0 for c in criteria}
	out: dict[str, float] = {}
	for c in criteria:
		out[f"avg_{c}"] = float(df[c].mean())
	out["avg_score"] = float(df[criteria].mean(axis=1).mean())
	return out


async def judge_vims(args: argparse.Namespace, client: AsyncOpenAI) -> None:
	predictions_path = Path(args.vims_predictions)
	gold_root = Path(args.vims_summary_dir)
	out_csv = Path(args.vims_output_csv)
	out_json = Path(args.vims_output_json)

	if not predictions_path.exists():
		raise FileNotFoundError(f"VIMS predictions not found: {predictions_path}")
	if not gold_root.exists():
		raise FileNotFoundError(f"VIMS summary dir not found: {gold_root}")

	predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
	if not isinstance(predictions, list):
		raise ValueError("VIMS predictions must be a JSON list.")

	semaphore = asyncio.Semaphore(args.judge_concurrency)
	criteria = ["groundness", "coherence", "completeness"]

	system_prompt = (
		"You are a strict evaluator for Vietnamese summaries. "
		"Score each criterion on a 1-5 integer scale. "
		"Return JSON only."
	)

	async def judge_one(item: dict[str, Any]) -> dict[str, Any]:
		cluster = str(item.get("cluster", ""))
		predicted_summary = str(item.get("summary", ""))
		gold_summary = find_vims_gold_summary(gold_root, cluster)

		if not gold_summary:
			log.warning("[VIMS] Missing gold summary for %s", cluster)
			return {
				"cluster": cluster,
				"category": cluster,
				"groundness": 1,
				"coherence": 1,
				"completeness": 1,
				"avg_score": 1.0,
				"predicted_summary": predicted_summary,
				"gold_summary": "",
			}

		user_prompt = (
			"Evaluate the model summary against the reference summary.\\n"
			"Criteria (1 to 5):\\n"
			"1) groundness: factual alignment with reference\\n"
			"2) coherence: clarity, flow, readability\\n"
			"3) completeness: coverage of key information\\n\\n"
			"Reference summary:\\n"
			f"{gold_summary}\\n\\n"
			"Model summary:\\n"
			f"{predicted_summary}\\n\\n"
			"Return strictly valid JSON with keys: "
			"groundness, coherence, completeness."
		)

		try:
			payload = await call_judge(
				client=client,
				semaphore=semaphore,
				model=args.judge_model,
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				max_tokens=args.judge_max_tokens,
				temperature=args.judge_temperature,
			)
		except Exception as exc:
			log.error("[VIMS] Judge call failed for %s: %s", cluster, exc)
			payload = {}

		g = clamp_score(payload.get("groundness"))
		c = clamp_score(payload.get("coherence"))
		p = clamp_score(payload.get("completeness"))
		avg_score = round((g + c + p) / 3.0, 4)

		return {
			"cluster": cluster,
			"category": cluster,
			"groundness": g,
			"coherence": c,
			"completeness": p,
			"avg_score": avg_score,
			"predicted_summary": predicted_summary,
			"gold_summary": gold_summary,
		}

	log.info("[VIMS] Judging %d clusters", len(predictions))
	results = await asyncio.gather(*(judge_one(item) for item in predictions))
	df = pd.DataFrame(results)

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_csv, index=False)

	report = {
		"task": "vims",
		"input_predictions": str(predictions_path),
		"input_gold_root": str(gold_root),
		"num_items": int(len(df)),
		"overall": summarize_criteria(df, criteria),
		# Category-based scoring for VIMS: each cluster is treated as a category.
		"category_scores": df[["category", "groundness", "coherence", "completeness", "avg_score"]].to_dict(
			orient="records"
		),
	}
	out_json.parent.mkdir(parents=True, exist_ok=True)
	out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	log.info("[VIMS] Saved row scores -> %s", out_csv)
	log.info("[VIMS] Saved report -> %s", out_json)


async def judge_vtsnlp(args: argparse.Namespace, client: AsyncOpenAI) -> None:
	predictions_path = Path(args.vtsnlp_predictions)
	out_csv = Path(args.vtsnlp_output_csv)
	out_json = Path(args.vtsnlp_output_json)

	if not predictions_path.exists():
		raise FileNotFoundError(f"VTSNLP predictions not found: {predictions_path}")

	df = pd.read_csv(predictions_path)
	required = {"instruct", "output", "llm_output", "category"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"VTSNLP CSV missing required columns: {sorted(missing)}")

	semaphore = asyncio.Semaphore(args.judge_concurrency)
	criteria = ["faithfulness", "coherence", "completeness", "instruction_following"]

	system_prompt = (
		"You are a strict evaluator for Vietnamese instruction-following answers. "
		"Score each criterion on a 1-5 integer scale. "
		"Return JSON only."
	)

	async def judge_one(idx: int, row: pd.Series) -> dict[str, Any]:
		user_prompt = (
			"Evaluate the candidate answer against the reference answer and instruction.\\n"
			"Criteria (1 to 5):\\n"
			"1) faithfulness: factual alignment with reference answer\\n"
			"2) coherence: clarity and fluency\\n"
			"3) completeness: coverage of key requested content\\n"
			"4) instruction_following: whether the instruction is obeyed\\n\\n"
			"Instruction:\\n"
			f"{row['instruct']}\\n\\n"
			"Reference answer:\\n"
			f"{row['output']}\\n\\n"
			"Candidate answer:\\n"
			f"{row['llm_output']}\\n\\n"
			"Return strictly valid JSON with keys: "
			"faithfulness, coherence, completeness, instruction_following."
		)

		try:
			payload = await call_judge(
				client=client,
				semaphore=semaphore,
				model=args.judge_model,
				system_prompt=system_prompt,
				user_prompt=user_prompt,
				max_tokens=args.judge_max_tokens,
				temperature=args.judge_temperature,
			)
		except Exception as exc:
			log.error("[VTSNLP] Judge call failed at row %d: %s", idx, exc)
			payload = {}

		f = clamp_score(payload.get("faithfulness"))
		c = clamp_score(payload.get("coherence"))
		p = clamp_score(payload.get("completeness"))
		i = clamp_score(payload.get("instruction_following"))
		avg_score = round((f + c + p + i) / 4.0, 4)

		return {
			"row_index": int(idx),
			"category": str(row["category"]),
			"faithfulness": f,
			"coherence": c,
			"completeness": p,
			"instruction_following": i,
			"avg_score": avg_score,
		}

	log.info("[VTSNLP] Judging %d rows", len(df))
	results = await asyncio.gather(*(judge_one(i, row) for i, row in df.iterrows()))
	judged_df = pd.DataFrame(results)

	merged = df.copy()
	score_columns = [
		"faithfulness",
		"coherence",
		"completeness",
		"instruction_following",
		"avg_score",
	]
	merged = merged.join(judged_df.set_index("row_index")[score_columns], how="left")

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(out_csv, index=False)

	by_category = (
		judged_df.groupby("category", as_index=False)[criteria + ["avg_score"]]
		.mean()
		.sort_values("category")
	)

	report = {
		"task": "vtsnlp",
		"input_predictions": str(predictions_path),
		"num_items": int(len(judged_df)),
		"overall": summarize_criteria(judged_df, criteria),
		"category_scores": by_category.to_dict(orient="records"),
	}
	out_json.parent.mkdir(parents=True, exist_ok=True)
	out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	log.info("[VTSNLP] Saved row scores -> %s", out_csv)
	log.info("[VTSNLP] Saved report -> %s", out_json)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="LLM-as-a-judge for VIMS and VTSNLP outputs.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument("--task", choices=["vims", "vtsnlp", "all"], default="all")
	parser.add_argument("--base_url", default="http://localhost:8000/v1", help="Judge model API base URL.")
	parser.add_argument("--judge_model", default="Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ")
	parser.add_argument("--api_key", default="EMPTY")
	parser.add_argument("--judge_concurrency", type=int, default=8)
	parser.add_argument("--judge_max_tokens", type=int, default=256)
	parser.add_argument("--judge_temperature", type=float, default=0.0)

	parser.add_argument(
		"--vims_predictions",
		default="runs/20260402_101721/vims_summaries/all_summaries.json",
		help="Path to VIMS predictions generated by vims_vllm.py",
	)
	parser.add_argument(
		"--vims_summary_dir",
		default="data/ViMs/summary",
		help="VIMS summary root containing Cluster_xxx/0.gold.txt",
	)
	parser.add_argument("--vims_output_csv", default="runs/judging/vims_judged_rows.csv")
	parser.add_argument("--vims_output_json", default="runs/judging/vims_judged_report.json")

	parser.add_argument(
		"--vtsnlp_predictions",
		default="runs/20260402_101721/vtsnlp_outputs.csv",
		help="Path to VTSNLP predictions generated by vtsnlp_vllm.py",
	)
	parser.add_argument("--vtsnlp_output_csv", default="runs/judging/vtsnlp_judged_rows.csv")
	parser.add_argument("--vtsnlp_output_json", default="runs/judging/vtsnlp_judged_report.json")

	return parser.parse_args()


async def main() -> None:
	args = parse_args()
	client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

	if args.task in {"vims", "all"}:
		await judge_vims(args, client)
	if args.task in {"vtsnlp", "all"}:
		await judge_vtsnlp(args, client)


if __name__ == "__main__":
	asyncio.run(main())
