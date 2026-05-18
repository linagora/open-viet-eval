# Potential integration

### 1. Change Detection

**Location:** `ingestion/pipeline.py` — `run()` method, lines 55-68

The pipeline performs incremental sync by comparing document versions:

1. Query all existing `Document` records for the current tenant + connector
2. Build a dict: `{source_id: source_version}` for known documents
3. Call `connector.list_changed_documents(known_versions)` which returns:
   - `new_or_changed`: documents where `source_version` differs or `source_id` is new
   - `deleted_ids`: `source_id`s present in DB but absent from source

This avoids re-processing unchanged documents entirely.

### 2. Multiple source document fetching (Connectors)

**Location:** `connectors/base.py`, `connectors/generic.py`

(filesystem, HTTP, SharePoint, Confluence, Elasticsearch)
#### Connector Interface

All connectors implement `BaseConnector`:

```python
class BaseConnector(ABC):
    def __init__(self, config: dict, credential: str = ""): ...
    def test_connection(self) -> bool: ...
    def list_documents(self) -> list[dict]: ...
    def fetch_document(self, source_id: str) -> RawDocument: ...
    def list_changed_documents(self, known_versions) -> tuple[list[dict], list[str]]: ...
```

Connectors are registered via the `@register_connector("name")` decorator and instantiated at runtime by `get_connector(connector_type, config, credential)`.

The output of `fetch_document()`:

```python
@dataclass
class RawDocument:
    source_id: str              # Unique ID in the source system
    title: str
    content: bytes | str        # Raw bytes (binary) or string (text/HTML)
    content_type: str           # MIME type or file extension
    source_url: str = ""
    author: str = ""
    path: str = ""
    doc_type: str = ""
    source_version: str = ""
    source_created_at: datetime | None = None
    source_modified_at: datetime | None = None
    metadata: dict = {}
```

## Analysis Overview

Analysis runs as a single Celery task (`analysis.tasks.run_unified_pipeline`) that executes seven phases sequentially:

```
AnalysisJob
  │
  ├─ Phase 1: Duplicate Detection        (analysis/duplicates.py)
  │            Multi-signal similarity + LLM verification
  │
  ├─ Phase 2: Claims Extraction           (analysis/claims.py)
  │            Extract atomic factual claims from chunks
  │
  ├─ Phase 3: Semantic Graph Construction (analysis/semantic_graph.py)  [optional]
  │            Build concept-level knowledge graph via NSG library
  │
  ├─ Phase 4: Topic Clustering            (analysis/clustering.py)
  │            HDBSCAN/KMeans on embeddings + LLM summaries
  │
  ├─ Phase 5: Gap Detection               (analysis/gaps.py)
  │            QG/RAG + orphan + stale + adjacent + structural graph analysis
  │
  ├─ Phase 6: Tree                        (inline in clustering)
  │            Hierarchical document taxonomy
  │
  └─ Phase 7: Contradiction Detection     (analysis/contradictions.py)
               Vector-search related claims + LLM classification
```
### 1. Duplicate Detection

**Location:** `analysis/duplicates.py` — `DuplicateDetector`
Finds duplicate and near-duplicate documents using a three-signal weighted scoring system with optional LLM verification

#### Algorithm

```
                    ┌─────────────────────────┐
                    │ Compute doc embeddings  │
                    │ (mean of chunk vectors) │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Build MinHash index     │
                    │ (3-word shingles,       │
                    │  128 permutations)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Find semantic candidates│
                    │ (cosine ≥ threshold×0.7)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Score each candidate    │     
                    │ pair (3 signals)        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Group by connected      │
                    │ components (BFS)        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ LLM verification        │
                    │ (high-scoring pairs)    │
                    └─────────────────────────┘
```
| Signal     | Weight | Method                                              |
|------------|--------|-----------------------------------------------------|
| **Semantic** | 0.55   | Cosine similarity of document embeddings             |
| **Lexical**  | 0.25   | MinHash Jaccard similarity (3-word shingles)         |
| **Metadata** | 0.20   | Average of: title similarity (SequenceMatcher), path similarity (SequenceMatcher), author match (1.0 or 0.0) |

Duplicate pairs are grouped using connected components via BFS:
- Each document = graph node
- Each qualifying pair = edge
- Connected components = duplicate groups

For pairs scoring above `cross_encoder_threshold` (default: 0.70):

1. Extract top 3 evidence chunks from each document (closest to each other by embedding)
2. Send to LLM with a structured verification prompt
3. LLM returns JSON:
   ```json
   {
     "classification": "duplicate" | "related" | "different",
     "confidence": 0.85,
     "evidence": "Both documents describe the same installation process..."
   }
   ```

**Group recommendation logic:**
- Any pair verified as `"duplicate"` → recommended action: `DELETE_OLDER`
- All pairs above `semantic_threshold` (0.92) but not verified → action: `REVIEW`
- Otherwise → action: `KEEP`

!! Did not detect some intentional duplication ?

### 2. Claims Extraction
**Location:** `analysis/contradictions.py` — `ClaimsExtractor`

Extracts atomic factual claims from document chunks for contradiction analysis.

#### Process

For each document chunk with content:

1. Send chunk text to LLM with the `CLAIMS_EXTRACTION` prompt
2. LLM returns structured JSON:
   ```json
   {
     "claims": [
       {
         "subject": "Free tier",
         "predicate": "allows",
         "object": "100 requests per minute",
         "qualifiers": {"as_of": "2024"},
         "raw_text": "Free tier: 100 requests per minute"
       }
     ]
   }
   ```
3. Create `Claim` records linked to the source document and chunk
4. Embed each claim's text representation via `llm.embed()`
5. Store claim vectors in `vec_claims` via `vec_store.upsert_claim()`

**Claim model fields:**
- `subject`: The entity the claim is about
- `predicate`: The relationship or action
- `object_value`: The value or target
- `qualifiers`: Additional context (dates, conditions) as JSON
- `raw_text`: Original text from which the claim was extracted

**Rate limiting:** Max `max_claims_per_chunk` (default: 5) claims per chunk to control cost.

=> Semantic Graph Construction

+ claim extraction is done for each chunk (might be too much)
+ strict claims extraction json format -> might be inflexible

### 4. Topic Clustering
**Location:** `analysis/clustering.py` — `TopicClusterEngine`

Discovers topic clusters from chunk embeddings using density-based or centroid-based clustering, projects to 2D for visualization, and generates LLM summaries.

#### Algorithm

```
Fetch all chunk vectors for tenant
          │
          ▼
    ┌─────────────┐
    │  HDBSCAN    │  (or KMeans fallback)
    │  clustering │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  PCA → 2D   │  (for visualization)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Create     │  TopicCluster records
    │  clusters   │  ClusterMembership records
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  LLM        │  Generate label + summary
    │  summaries  │  per cluster
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Build tree │  TreeNode hierarchy
    └─────────────┘
```
### Graph-Augmented RAG (optional)

When `semantic_graph.enabled` is `true` and a persisted graph exists for the project:

1. After vector search, load the project graph via `analysis.semantic_graph.load_graph()`
2. Run `nsg.query_subgraph(question)` — finds the top 5 seed concepts closest to the query, then expands 1 hop to collect related concepts and edges (max 20 nodes)
3. Format the concept context: seed concept names + relationship triples (`src —[relation]→ dst`)
4. Prepend the concept context to the chunk context in the system prompt

This provides the LLM with a structural understanding of how concepts relate, complementing the raw text passages from vector search. For example, if a user asks about "rate limits", the graph might surface related concepts like "quotas", "throttling", "API tiers" with their relationships, even if those terms don't appear in the top vector search results.

The graph augmentation fails gracefully — if the graph file doesn't exist or loading fails, standard RAG continues without it.

### Others
- Gap detection
- Contradict detection

# SCORE Formula

SCORE is a Nutri-Score-style quality grade (A through E) for a knowledge base. It evaluates the overall health, consistency, and completeness of a document repository by combining metrics from the latest completed LLM analysis **and** the latest RAG audit (if available).

## How It Works

### LLM Analysis Score (5 dimensions)

The base score starts at **100** and penalties are subtracted across five dimensions. Each dimension has a maximum penalty, totalling 100 points.

```
llm_score = 100 - uniqueness_penalty
                - consistency_penalty
                - coverage_penalty
                - structure_penalty
                - health_penalty
```

The LLM score is clamped to `[0, 100]`.

### Composite Score (LLM + RAG Audit)

When both an LLM analysis and a RAG audit are completed, the final SCORE is a **weighted composite**:

```
final_score = llm_score × 0.85 + audit_rag_score × 0.15
```

The composite is clamped to `[0, 100]` and mapped to a letter grade.

---

## Part 1 — LLM Analysis Dimensions

### 1. Uniqueness (max penalty: 20 points)

Measures how free the repository is from duplicate content.

**Inputs:**
- `actionable_dup_groups`: Number of `DuplicateGroup` records from the latest analysis, excluding groups with `recommended_action = "keep"` (these are related but distinct documents).
- `total_docs`: Total non-deleted documents in the tenant.

**Formula:**
```
dup_ratio = actionable_dup_groups / total_docs
uniqueness_penalty = min(20, dup_ratio / 0.30 * 20)
```

A duplicate ratio of **30% or higher** triggers the full 20-point penalty. Zero duplicates means zero penalty.

**Sub-score:** `uniqueness = 100 - (uniqueness_penalty / 20 * 100)`

---

### 2. Consistency (max penalty: 25 points)

Measures how free the repository is from contradictory or outdated information.

**Inputs:**
- Contradiction pairs from the latest analysis where `classification` is `"contradiction"` or `"outdated"` (taken from LLM result of claim extraction), grouped by severity:
  - `high_c`: Count with severity `"high"`
  - `med_c`: Count with severity `"medium"`
  - `low_c`: Count with severity `"low"`

**Formula:**
```
weighted_contradictions = high_c * 3 + med_c * 2 + low_c * 1
contra_ratio = weighted_contradictions / total_docs
consistency_penalty = min(25, contra_ratio / 0.50 * 25)
```

Severity weights reflect impact: a single high-severity contradiction counts 3x more than a low-severity one. A weighted ratio of **0.50 or higher** triggers the full 25-point penalty.

**Sub-score:** `consistency = 100 - (consistency_penalty / 25 * 100)`

---

### 3. Coverage (max penalty: 25 points)

Measures how complete the knowledge base is, based on detected gaps.

**Inputs:**
- Gap reports from the latest analysis (types: `missing_topic`, `low_coverage`, `stale_area`, `orphan_topic`, `weak_bridge`, `concept_island`), grouped by severity:
  - `high_g`, `med_g`, `low_g`: Counts per severity level
- `avg_coverage_score`: Average `coverage_score` across all gap reports (0-1 scale, where lower = bigger gap)

**Formula (two components):**
```
# Component A: Gap count penalty (max 15)
weighted_gaps = high_g * 3 + med_g * 2 + low_g * 1
gap_ratio = weighted_gaps / total_docs
gap_penalty = min(15, gap_ratio / 0.50 * 15)

# Component B: Coverage depth penalty (max 10)
if avg_coverage_score is available:
    coverage_adj = (1 - avg_coverage_score) * 10
else:
    coverage_adj = 5   # assume moderate gaps when no data

# Combined
coverage_penalty = min(25, gap_penalty + coverage_adj)
```

Component A penalizes the sheer number of gaps (weighted by severity). Component B penalizes low average coverage depth. Together they cap at 25 points.

**Sub-score:** `coverage = 100 - (coverage_penalty / 25 * 100)`

---

### 4. Structure (max penalty: 15 points)

Measures how well-organized the content is into coherent topic clusters.

**Inputs:**
- `avg_cohesion`: Average `similarity_to_centroid` across all `ClusterMembership` records for the latest analysis (0-1 scale, higher = tighter clusters).
- `cluster_count`: Number of `TopicCluster` records.

**Formula:**
```
structure_penalty = 0

if avg_cohesion is available:
    structure_penalty += max(0, (1 - avg_cohesion)) * 10    # max ~10
else:
    structure_penalty += 8   # no cohesion data available

if cluster_count == 0:
    structure_penalty += 5   # no clusters detected at all

structure_penalty = min(15, structure_penalty)
```

High cohesion (e.g. 0.85) means documents within each cluster are tightly related, resulting in a small penalty of ~1.5. Low cohesion (e.g. 0.40) means clusters are loose and overlapping, resulting in a penalty of ~6.

**Sub-score:** `structure = 100 - (structure_penalty / 15 * 100)`

---

### 5. Health (max penalty: 15 points)

Measures the operational state of the document pipeline.

**Inputs:**
- `ready_docs`: Documents with status `"ready"` (fully ingested, chunked, and embedded).
- `error_docs`: Documents with status `"error"`.
- `total_docs`: Total non-deleted documents.

**Health sub-score (0-100):**
```
error_penalty = min(50, (error_docs / total_docs) * 500)
ready_bonus = (ready_docs / total_docs) * 100
health = clamp(0, 100, round(ready_bonus - error_penalty))
```

An error rate of 10% triggers a 50-point health penalty. 100% readiness with 0% errors gives a perfect 100.

**Mapped to main score:**
```
health_penalty = (100 - health) / 100 * 15
```

**Sub-score:** `health` (the 0-100 value directly)

---

## Part 2 — RAG Audit (6th Dimension: "Qualité RAG")

The RAG audit is a **fully automated, LLM-free** evaluation that runs 6 axes sequentially via a Celery task (`analysis.audit.runner.run_audit`). Each axis produces a score (0-100), and the overall audit score is a **weighted average** of all axis scores:

```
overall_audit_score = Σ(axis_score × axis_weight) / Σ(axis_weight)
```

Default axis weight: `1/6` per axis (uniform). Custom weights can be set in `config.yaml` under `audit.axis_weights`.

The overall audit score is mapped to a letter grade using the same A-E scale as SCORE.

---

### Axis 1 — Hygiène du corpus (Corpus Hygiene)

**Purpose:** Detects exact duplicates, near-duplicates, boilerplate content, language fragmentation, and PII/secrets exposure.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Exact dedup | SHA-256 `content_hash` counting via `collections.Counter` | Python stdlib (`collections`, `hashlib`) | — |
| Near-duplicate | MinHash LSH (Locality-Sensitive Hashing) | **`datasketch`** (`MinHash`, `MinHashLSH`) | `num_perm=128`, shingle size: 3-word, Jaccard threshold: `0.5`, sample cap: 2000 chunks |
| Boilerplate | Normalized line frequency analysis (line appears in >N% of documents) | Python stdlib (`collections.Counter`) | Frequency threshold: `0.30` (30% of docs), min line length: 10 chars |
| Language detection | Statistical language classifier | **`langid`** (`langid.classify`) | Sample: 200 chunks, first 500 chars per chunk, min 20 chars |
| PII / secrets | Regular expression pattern matching | Python stdlib (`re`) | 6 patterns: email, phone_fr, phone_intl, api_key, ip_address, secret_generic; sample cap: 1000 chunks |

**Scoring formula:**
```
uniqueness_score  = max(0, 100 × (1 - exact_dup_ratio × 5))
neardup_score     = max(0, 100 × (1 - neardup_ratio × 3))
boilerplate_score = max(0, 100 × (1 - boilerplate_ratio × 3))
pii_score         = max(0, 100 × (1 - pii_ratio × 10))
lang_score        = min(100, dominant_language_ratio × 100)

hygiene_score = 0.30 × uniqueness_score
              + 0.20 × neardup_score
              + 0.20 × boilerplate_score
              + 0.15 × lang_score
              + 0.15 × pii_score
```

---

### Axis 2 — Structure RAG (RAG Structure)

**Purpose:** Evaluates chunk sizing uniformity, information density, readability, and inter-chunk overlap.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Size uniformity | Coefficient of Variation (CV = std / mean) | Python stdlib (`math.sqrt`) | min_tokens: `50`, max_tokens: `1024`, optimal: `512` |
| Outlier detection | Count of chunks below `min_tokens` or above `max_tokens` | — | Penalty multiplier: `×3` on outlier_ratio |
| Info density | Stopword ratio (1 - stopwords/total_words) | Python stdlib (`re`) | Combined FR + EN stopword set (~150 words), word extraction via `\w+` regex |
| Readability | Sentences/chunk and words/sentence heuristics | Python stdlib (`re.split`) | Sentence split on `[.!?]+`, penalty if avg words/sentence > 30 or avg sentences < 2 |
| Overlap | Jaccard similarity on token sets between consecutive same-document chunks | Python stdlib (`set` operations) | Token extraction via `\w+` regex |

**Scoring formula:**
```
cv = std_tokens / mean_tokens
uniformity_score  = max(0, 100 × (1 - cv))
outlier_score     = max(0, 100 × (1 - outlier_ratio × 3))
density_score     = min(100, avg_density × 150)
readability_score = 100 - penalties (words/sentence > 30: -2 per excess word, cap 40; sentences < 2: -20)

structure_score = 0.30 × uniformity_score
                + 0.25 × outlier_score
                + 0.25 × density_score
                + 0.20 × readability_score
```

**Visualizations:** Token histogram (25 bins), box plot per source (Q1/median/Q3/min/max), scatter plot (tokens vs density, up to 500 points).

---

### Axis 3 — Couverture sémantique (Semantic Coverage)

**Purpose:** Measures topic diversity, balance, and semantic coverage of the corpus using unsupervised NLP.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Vectorization | TF-IDF (Term Frequency - Inverse Document Frequency) | **`scikit-learn`** (`TfidfVectorizer`) | `max_features=10000`, `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95` |
| Dimensionality reduction | Truncated SVD (Latent Semantic Analysis / LSA) | **`scikit-learn`** (`TruncatedSVD`) | `n_components=50` (capped at `min(50, n_features-1, n_docs-1)`, min 2), `random_state=42` |
| Normalization | L2 normalization on SVD vectors | **`scikit-learn`** (`sklearn.preprocessing.normalize`) | — |
| Topic modeling | NMF (Non-Negative Matrix Factorization) | **`scikit-learn`** (`NMF`) | `k = max(3, min(√(n_docs/2), 20))`, `max_iter=300`, `random_state=42`, top 10 terms per topic |
| Clustering | KMeans on L2-normalized SVD vectors | **`scikit-learn`** (`KMeans`) | `n_clusters=k_topics`, `n_init=10`, `random_state=42` |
| Outlier detection | Local Outlier Factor (LOF) | **`scikit-learn`** (`LocalOutlierFactor`) | `contamination=0.05`, `n_neighbors=min(20, n_docs-1)`, requires `n_docs >= 20` |
| Topic balance | Gini coefficient on cluster sizes | Custom implementation | `gini = (2 × Σ(i+1)×v_i) / (n × Σv_i) - (n+1)/n` |
| 2D projection | PCA (Principal Component Analysis) | **`scikit-learn`** (`PCA`) | `n_components=2`, `random_state=42`, applied on normalized SVD vectors |

**Pipeline:**
1. TF-IDF vectorization (uni+bigrams) → sparse matrix
2. TruncatedSVD (50 components) → dense matrix for clustering
3. L2 normalization → unit vectors
4. NMF topic modeling on TF-IDF matrix → topic-document matrix + topic terms
5. KMeans clustering on normalized SVD vectors → cluster assignments
6. LOF on normalized SVD vectors → outlier labels (if n_docs >= 20)
7. PCA 2D on normalized SVD vectors → scatter coordinates
8. Gini coefficient on cluster size distribution

**Scoring formula:**
```
balance_score   = (1 - gini_coefficient) × 100
coverage_score  = (covered_topics / k_topics) × 100           # covered = topics with ≥ 3 docs
outlier_score   = max(0, (1 - outlier_ratio × 5)) × 100
coherence_score = avg_intra_cluster_cosine_similarity × 100   # centroid dot product

coverage_axis_score = 0.30 × balance_score
                    + 0.30 × coverage_score
                    + 0.20 × outlier_score
                    + 0.20 × coherence_score
```

---

### Axis 4 — Cohérence interne (Internal Coherence)

**Purpose:** Detects terminology variants, key-value conflicts, and entity inconsistencies across the corpus.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Term extraction | TF-IDF per-document, top-20 terms | **`scikit-learn`** (`TfidfVectorizer`) | `max_features=5000`, `ngram_range=(1, 1)`, `min_df=1`, `max_df=0.95` |
| Variant detection (stemming) | Snowball stemmer (French) | **`nltk`** (`nltk.stem.snowball.SnowballStemmer`) | Language: `"french"` |
| Variant detection (similarity) | SequenceMatcher ratio | Python stdlib (`difflib.SequenceMatcher`) | Similarity threshold: `0.85` |
| KV conflict detection | Regex extraction for 7 key types | Python stdlib (`re`) | Keys: `sla`, `version`, `port`, `url`, `date`, `timeout`, `limit` |
| Entity consistency | Regex extraction for 4 entity types | Python stdlib (`re`) | Types: `date`, `version`, `url`, `ip` |

**Variant detection algorithm:**
1. Extract top-20 TF-IDF terms per document
2. Group all terms by their French Snowball stem
3. For groups with ≥2 surface forms, check pairwise `SequenceMatcher.ratio()`
4. If any pair has ratio ≥ 0.85 and different surface forms → variant group
5. Canonical form = most frequent surface form across documents

**KV conflict detection:**
Regex patterns extract structured values (e.g., `SLA := 99.9%`, `port = 8080`, `timeout = 30s`) from all chunks. Values are normalized (strip + lowercase). A conflict exists when the same key has multiple distinct values across different documents.

**Scoring formula:**
```
conflict_ratio = total_conflicting_values / total_docs
conflict_score = max(0, 100 × (1 - conflict_ratio × 5))

term_consistency = max(0, 100 - variant_groups × 2)

entity_conflict_count = Σ(len(values) - 1) for entities with > 1 value
entity_score = max(0, 100 × (1 - entity_conflict_count / (total_docs × 3)))

coherence_score = 0.40 × conflict_score
                + 0.30 × term_consistency
                + 0.30 × entity_score
```

---

### Axis 5 — Retrievability

**Purpose:** Evaluates how well documents can be found using full-text search, by building a BM25 index and running auto-generated queries against it.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Full-text index | BM25 Okapi | **`rank-bm25`** (`BM25Okapi`) | Tokenization: `\w+` regex on lowercased content |
| Query generation (titles) | Direct document title usage | — | — |
| Query generation (headings) | Direct heading path usage | — | Min length: 3 chars |
| Query generation (bigrams) | TF-IDF bigram extraction per document | **`scikit-learn`** (`TfidfVectorizer`) | `ngram_range=(2, 2)`, `max_features=5000`, `min_df=1`, `max_df=0.9`, top N per doc |
| MRR | Mean Reciprocal Rank | Custom | `MRR = avg(1/rank_of_expected_doc)` |
| Recall@k | Recall at k values | Custom | k = `[1, 3, 5, 10, 20]` |
| Diversity | Unique doc ratio in top-10 results | Custom | `diversity = |unique_docs_in_top10| / total_docs` |

**Pipeline:**
1. Tokenize all chunks (lowercased, `\w+` regex) → build `BM25Okapi` index
2. Generate queries:
   - Document titles as queries
   - Chunk heading paths as queries
   - Top TF-IDF bigrams per document (configurable `queries_per_doc`, default 3)
   - Deduplicate and limit to 500 queries
3. For each query, score against BM25 index, rank results
4. Evaluate: find rank of expected document in results
5. Compute MRR, Recall@k, zero-result ratio, diversity

**Scoring formula:**
```
retrievability_score = 0.35 × MRR × 100
                     + 0.30 × Recall@10 × 100
                     + 0.20 × (1 - zero_result_ratio) × 100
                     + 0.15 × min(diversity, 1.0) × 100
```

---

### Axis 6 — Gouvernance & metadata (Governance & Metadata)

**Purpose:** Evaluates metadata completeness, document freshness, orphan documents, and path-based connectivity.

**Algorithms and libraries:**

| Sub-metric | Algorithm | Library | Parameters |
|---|---|---|---|
| Metadata completeness | Field fill-rate across required fields | — | Required fields: `["author", "source_modified_at", "doc_type", "path"]` |
| Staleness | Age computation from `source_modified_at` (or `created_at` fallback) | Python stdlib (`datetime.timedelta`) | Threshold: `180` days |
| Orphan detection | Documents with no `path` AND no `source_url` | — | — |
| Path connectivity | Shared path prefix grouping (first 2 path segments) | Python stdlib (`collections.defaultdict`) | Connectivity = fraction of docs in groups > 1 |
| Per-source completeness | Aggregate fill-rate by connector source | — | — |

**Scoring formula:**
```
completeness_score = avg_field_fill_rate × 100

freshness_score = max(0, (1 - stale_ratio) × 100)

orphan_score = max(0, (1 - orphan_ratio × 3) × 100)

connectivity_score:
  - No edges between path prefix groups → 30.0
  - Otherwise: min(100, (docs_in_multi_doc_groups / total_docs_with_paths) × 100)

governance_score = 0.30 × completeness_score
                 + 0.25 × freshness_score
                 + 0.25 × orphan_score
                 + 0.20 × connectivity_score
```
