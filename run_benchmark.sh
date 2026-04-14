#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "unicorn-team/Unicorn-R3"
  "QuantTrio/Qwen3.5-9B-AWQ"
  "Qwen/Qwen3-8B-AWQ"
)
DEFAULT_JUDGE_MODEL="QuantTrio/Qwen3.5-9B-AWQ"
IMAGE="vllm/vllm-openai:latest"
CONTAINER_NAME="vllm_qwen3"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-80}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/vllm_test/.venv/bin/python}"

BASE_URL="http://localhost:${PORT}/v1"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT_BASE="${RESULT_ROOT:-./runs/${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT_BASE}"
LOG_DIR="${RESULT_ROOT}/logs"
VLLM_LOG_FILE="${LOG_DIR}/vllm_container.log"
VLLM_LOG_PID=""

VIMS_INPUT_DIR="${VIMS_INPUT_DIR:-data/ViMs/S3_summary}"
VIMS_OUTPUT_DIR="${VIMS_OUTPUT_DIR:-${RESULT_ROOT}/vims_summaries}"
VIMS_CONCURRENCY="${VIMS_CONCURRENCY:-80}"
VIMS_MAX_TOKENS="${VIMS_MAX_TOKENS:-256}"
VIMS_TEMPERATURE="${VIMS_TEMPERATURE:-0.3}"
VIMS_SAMPLE_RATE="${VIMS_SAMPLE_RATE:-1}"

VLMU_INPUT_FILE="${VLMU_INPUT_FILE:-data/test.jsonl}"
VLMU_OUTPUT_FILE="${VLMU_OUTPUT_FILE:-${RESULT_ROOT}/vlmu_results.csv}"
VLMU_CONCURRENCY="${VLMU_CONCURRENCY:-80}"
VLMU_MAX_TOKENS="${VLMU_MAX_TOKENS:-5}"
VLMU_TEMPERATURE="${VLMU_TEMPERATURE:-0.0}"
VLMU_SAVE_EVERY="${VLMU_SAVE_EVERY:-50}"
VLMU_SAMPLE_RATE="${VLMU_SAMPLE_RATE:-1}"

VTSNLP_INPUT_FILE="${VTSNLP_INPUT_FILE:-data/filtered_data.parquet}"
VTSNLP_OUTPUT_FILE="${VTSNLP_OUTPUT_FILE:-${RESULT_ROOT}/vtsnlp_outputs.csv}"
VTSNLP_CONCURRENCY="${VTSNLP_CONCURRENCY:-80}"
VTSNLP_MAX_TOKENS="${VTSNLP_MAX_TOKENS:-512}"
VTSNLP_TEMPERATURE="${VTSNLP_TEMPERATURE:-0.7}"
VTSNLP_SAVE_EVERY="${VTSNLP_SAVE_EVERY:-100}"
VTSNLP_SAMPLE_RATE="${VTSNLP_SAMPLE_RATE:-0.005}"

SQUAD_INPUT_FILE="${SQUAD_INPUT_FILE:-data/validation.csv}"
SQUAD_OUTPUT_FILE="${SQUAD_OUTPUT_FILE:-${RESULT_ROOT}/squad_results.csv}"
SQUAD_CONCURRENCY="${SQUAD_CONCURRENCY:-80}"
SQUAD_MAX_TOKENS="${SQUAD_MAX_TOKENS:-128}"
SQUAD_TEMPERATURE="${SQUAD_TEMPERATURE:-0.0}"
SQUAD_SAMPLE_RATE="${SQUAD_SAMPLE_RATE:-1}"

JUDGE_SCRIPT="${JUDGE_SCRIPT:-judge.py}"
JUDGE_TASK="${JUDGE_TASK:-all}"
JUDGE_MODEL="QuantTrio/Qwen3.5-9B-AWQ"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-80}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-128}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_VIMS_SUMMARY_DIR="${JUDGE_VIMS_SUMMARY_DIR:-data/ViMs/summary}"

EVAL_SCRIPTS=("vims_vllm.py" "vlmu_vllm.py" "vtsnlp_vllm.py" "squad_vllm.py")

cleanup() {
  if [[ -n "${VLLM_LOG_PID}" ]]; then
    kill "${VLLM_LOG_PID}" >/dev/null 2>&1 || true
  fi
  echo "[CLEANUP] Stopping container: ${CONTAINER_NAME}"
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

start_vllm_log_stream() {
  sudo docker logs -f "${CONTAINER_NAME}" >> "${VLLM_LOG_FILE}" 2>&1 &
  VLLM_LOG_PID=$!
}

stop_vllm_log_stream() {
  if [[ -n "${VLLM_LOG_PID}" ]]; then
    kill "${VLLM_LOG_PID}" >/dev/null 2>&1 || true
    VLLM_LOG_PID=""
  fi
}

start_vllm_container() {
  local model="$1"
  local model_upper
  model_upper="${model^^}"

  if [[ "${model_upper}" == *"AWQ"* ]]; then
    echo "[INFO] AWQ model detected; starting vLLM with --quantization awq"
    sudo docker run -d \
      --name "${CONTAINER_NAME}" \
      --runtime nvidia \
      --gpus all \
      -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
      -p "${PORT}:8000" \
      --ipc=host \
      "${IMAGE}" \
      --model "${model}" \
      --quantization awq \
      --trust-remote-code \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --default-chat-template-kwargs '{"enable_thinking": false}'
  else
    sudo docker run -d \
      --name "${CONTAINER_NAME}" \
      --runtime nvidia \
      --gpus all \
      -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
      -p "${PORT}:8000" \
      --ipc=host \
      "${IMAGE}" \
      --model "${model}" \
      --trust-remote-code \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --default-chat-template-kwargs '{"enable_thinking": false}'
  fi
}

run_eval() {
  local name="$1"
  local script="$2"
  shift 2

  local log_file="${LOG_DIR}/${name}.log"
  echo "[RUN] ${script}"
  echo "[LOG] ${log_file}"
  "${PYTHON_BIN}" "${script}" "$@" 2>&1 | tee "${log_file}"
  echo "[DONE] ${script}"
}

wait_for_server() {
  local sleep_seconds=3

  echo "[WAIT] Waiting for vLLM API at http://localhost:${PORT}/v1/models ..."
  while true; do
    if curl -sS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[OK] vLLM server is ready."
      return 0
    fi

    if ! sudo docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
      echo "[ERROR] vLLM container is not running. Check startup logs in ${VLLM_LOG_FILE}."
      return 1
    fi

    sleep "${sleep_seconds}"
  done
}

main() {
  trap cleanup EXIT

  # Store per-model outputs for final judging pass.
  local -a MODEL_SLUGS=()
  local -a MODEL_RESULT_ROOTS=()
  local -a MODEL_VIMS_PRED_FILES=()
  local -a MODEL_VTSNLP_PRED_FILES=()

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python interpreter not found or not executable: ${PYTHON_BIN}"
    exit 1
  fi

  for script in "${EVAL_SCRIPTS[@]}"; do
    if [[ ! -f "${script}" ]]; then
      echo "[ERROR] Missing evaluation script: ${script}"
      exit 1
    fi
  done

  if [[ ! -f "${JUDGE_SCRIPT}" ]]; then
    echo "[ERROR] Missing judge script: ${JUDGE_SCRIPT}"
    exit 1
  fi

  mkdir -p "${HF_CACHE_DIR}"

  for MODEL in "${MODELS[@]}"; do
    model_slug="${MODEL##*/}"
    model_slug="${model_slug//./_}"

    RESULT_ROOT="${RESULT_ROOT_BASE}/${model_slug}"
    LOG_DIR="${RESULT_ROOT}/logs"
    VLLM_LOG_FILE="${LOG_DIR}/vllm_container.log"
    VIMS_OUTPUT_DIR="${RESULT_ROOT}/vims_summaries"
    VLMU_OUTPUT_FILE="${RESULT_ROOT}/vlmu_results_${model_slug}.csv"
    VTSNLP_OUTPUT_FILE="${RESULT_ROOT}/vtsnlp_outputs.csv"
    SQUAD_OUTPUT_FILE="${RESULT_ROOT}/squad_results.csv"
    JUDGE_VIMS_PREDICTIONS="${VIMS_OUTPUT_DIR}/all_summaries.json"
    JUDGE_VIMS_OUTPUT_CSV="${RESULT_ROOT}/judging/vims_judged_rows.csv"
    JUDGE_VIMS_OUTPUT_JSON="${RESULT_ROOT}/judging/vims_judged_report.json"
    JUDGE_VTSNLP_PREDICTIONS="${VTSNLP_OUTPUT_FILE}"
    JUDGE_VTSNLP_OUTPUT_CSV="${RESULT_ROOT}/judging/vtsnlp_judged_rows.csv"
    JUDGE_VTSNLP_OUTPUT_JSON="${RESULT_ROOT}/judging/vtsnlp_judged_report.json"

    mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${VIMS_OUTPUT_DIR}"

    cat > "${RESULT_ROOT}/run_config.txt" <<EOF
RUN_TAG=${RUN_TAG}
RESULT_ROOT=${RESULT_ROOT}
MODEL=${MODEL}
BASE_URL=${BASE_URL}
VIMS_OUTPUT_DIR=${VIMS_OUTPUT_DIR}
VLMU_OUTPUT_FILE=${VLMU_OUTPUT_FILE}
VTSNLP_OUTPUT_FILE=${VTSNLP_OUTPUT_FILE}
SQUAD_OUTPUT_FILE=${SQUAD_OUTPUT_FILE}
JUDGE_TASK=${JUDGE_TASK}
JUDGE_MODEL=${JUDGE_MODEL}
JUDGE_VIMS_PREDICTIONS=${JUDGE_VIMS_PREDICTIONS}
JUDGE_VIMS_SUMMARY_DIR=${JUDGE_VIMS_SUMMARY_DIR}
JUDGE_VIMS_OUTPUT_CSV=${JUDGE_VIMS_OUTPUT_CSV}
JUDGE_VIMS_OUTPUT_JSON=${JUDGE_VIMS_OUTPUT_JSON}
JUDGE_VTSNLP_PREDICTIONS=${JUDGE_VTSNLP_PREDICTIONS}
JUDGE_VTSNLP_OUTPUT_CSV=${JUDGE_VTSNLP_OUTPUT_CSV}
JUDGE_VTSNLP_OUTPUT_JSON=${JUDGE_VTSNLP_OUTPUT_JSON}
EOF

    echo "[INFO] Evaluating model: ${MODEL}"
    echo "[INFO] Removing any existing container named ${CONTAINER_NAME} ..."
    sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    echo "[INFO] Starting vLLM container..."
    start_vllm_container "${MODEL}"

    echo "[INFO] Streaming vLLM startup logs (also saved to ${VLLM_LOG_FILE})"
    start_vllm_log_stream

    wait_for_server
    stop_vllm_log_stream
    echo "[INFO] vLLM startup log saved: ${VLLM_LOG_FILE}"

    run_eval "01_vims" "vims_vllm.py" \
      --input_dir "${VIMS_INPUT_DIR}" \
      --output_dir "${VIMS_OUTPUT_DIR}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency "${VIMS_CONCURRENCY}" \
      --max_tokens "${VIMS_MAX_TOKENS}" \
      --temperature "${VIMS_TEMPERATURE}" \
      --sample_rate "${VIMS_SAMPLE_RATE}" \
      --suppress_thinking

    run_eval "02_vlmu" "vlmu_vllm.py" \
      --input_file "${VLMU_INPUT_FILE}" \
      --output_file "${VLMU_OUTPUT_FILE}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency "${VLMU_CONCURRENCY}" \
      --max_tokens "${VLMU_MAX_TOKENS}" \
      --temperature "${VLMU_TEMPERATURE}" \
      --save_every "${VLMU_SAVE_EVERY}" \
      --sample_rate "${VLMU_SAMPLE_RATE}" \
      --suppress_thinking

    run_eval "03_vtsnlp" "vtsnlp_vllm.py" \
      --input_file "${VTSNLP_INPUT_FILE}" \
      --output_file "${VTSNLP_OUTPUT_FILE}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency "${VTSNLP_CONCURRENCY}" \
      --max_tokens "${VTSNLP_MAX_TOKENS}" \
      --temperature "${VTSNLP_TEMPERATURE}" \
      --save_every "${VTSNLP_SAVE_EVERY}" \
      --sample_rate "${VTSNLP_SAMPLE_RATE}" \
      --instruct_mode

    run_eval "04_squad" "squad_vllm.py" \
      --input_file "${SQUAD_INPUT_FILE}" \
      --output_file "${SQUAD_OUTPUT_FILE}" \
      --base_url "${BASE_URL}" \
      --model "${MODEL}" \
      --concurrency "${SQUAD_CONCURRENCY}" \
      --max_tokens "${SQUAD_MAX_TOKENS}" \
      --temperature "${SQUAD_TEMPERATURE}" \
      --sample_rate "${SQUAD_SAMPLE_RATE}" \
      --suppress_thinking

    MODEL_SLUGS+=("${model_slug}")
    MODEL_RESULT_ROOTS+=("${RESULT_ROOT}")
    MODEL_VIMS_PRED_FILES+=("${JUDGE_VIMS_PREDICTIONS}")
    MODEL_VTSNLP_PRED_FILES+=("${JUDGE_VTSNLP_PREDICTIONS}")
  done

  echo "[INFO] Starting final judging phase with judge model: ${JUDGE_MODEL}"
  echo "[INFO] Restarting vLLM for judge model..."
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  RESULT_ROOT="${RESULT_ROOT_BASE}/judge"
  LOG_DIR="${RESULT_ROOT}/logs"
  VLLM_LOG_FILE="${LOG_DIR}/vllm_container.log"
  mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

  start_vllm_container "${JUDGE_MODEL}"

  start_vllm_log_stream
  wait_for_server
  stop_vllm_log_stream

  for i in "${!MODEL_SLUGS[@]}"; do
    model_slug="${MODEL_SLUGS[$i]}"
    model_result_root="${MODEL_RESULT_ROOTS[$i]}"
    JUDGE_VIMS_PREDICTIONS="${MODEL_VIMS_PRED_FILES[$i]}"
    JUDGE_VTSNLP_PREDICTIONS="${MODEL_VTSNLP_PRED_FILES[$i]}"
    JUDGE_VIMS_OUTPUT_CSV="${model_result_root}/judging/vims_judged_rows.csv"
    JUDGE_VIMS_OUTPUT_JSON="${model_result_root}/judging/vims_judged_report.json"
    JUDGE_VTSNLP_OUTPUT_CSV="${model_result_root}/judging/vtsnlp_judged_rows.csv"
    JUDGE_VTSNLP_OUTPUT_JSON="${model_result_root}/judging/vtsnlp_judged_report.json"

    LOG_DIR="${model_result_root}/logs"
    mkdir -p "${LOG_DIR}" "${model_result_root}/judging"

    run_eval "05_judge_${model_slug}" "${JUDGE_SCRIPT}" \
      --task "${JUDGE_TASK}" \
      --base_url "${BASE_URL}" \
      --judge_model "${JUDGE_MODEL}" \
      --api_key "${JUDGE_API_KEY}" \
      --judge_concurrency "${JUDGE_CONCURRENCY}" \
      --judge_max_tokens "${JUDGE_MAX_TOKENS}" \
      --judge_temperature "${JUDGE_TEMPERATURE}" \
      --vims_predictions "${JUDGE_VIMS_PREDICTIONS}" \
      --vims_summary_dir "${JUDGE_VIMS_SUMMARY_DIR}" \
      --vims_output_csv "${JUDGE_VIMS_OUTPUT_CSV}" \
      --vims_output_json "${JUDGE_VIMS_OUTPUT_JSON}" \
      --vtsnlp_predictions "${JUDGE_VTSNLP_PREDICTIONS}" \
      --vtsnlp_output_csv "${JUDGE_VTSNLP_OUTPUT_CSV}" \
      --vtsnlp_output_json "${JUDGE_VTSNLP_OUTPUT_JSON}"
  done

  echo "[DONE] All evaluation and judging scripts completed successfully."
  echo "[DONE] Results root directory: ${RESULT_ROOT_BASE}"
}

main "$@"
