# VERITASAI: SYSTEM REQUIREMENTS SPECIFICATION & UX DESIGN ARCHITECTURE
**Version 1.0 — Production-Ready Specifications**

---

## MODULE 1: Executive Summary & Core Architectural Vision

### 1.1 Project Overview
**VeritasAI** (derived from the Latin *Veritas*, meaning "truth") is a production-grade, multi-LLM "truth engine" designed to combat artificial intelligence hallucinations. Operating on the core principle that **"no single LLM is always right,"** VeritasAI queries multiple independent model providers in parallel, subjecting their outputs to a dual-signal verification pipeline:
1. **A Mathematical Signal:** Semantic embedding extraction and spatial clustering (DBSCAN + Cosine Similarity).
2. **A Social Signal:** Anonymized, cross-model peer review (models grading other models).

By synthesizing these signals, VeritasAI generates a single, high-confidence consolidated response alongside individual trust scores, hallucination warnings, and an audit trail of raw model answers.

```
                    +-----------------------+
                    |  Raw User Query (UI)  |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    | Stage 1: Enhancer     | (Groq Llama 3.1 8B)
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    | Stage 2: Parallel     | (Up to 7 Parallel LLMs)
                    |          Dispatcher   |
                    +-----+-----+-----+-----+
                          |     |     |
            +-------------+     |     +-------------+
            v                   v                   v
      +-----+------+      +-----+------+      +-----+------+
      | LLM Result |      | LLM Result |      | LLM Result |
      +-----+------+      +-----+------+      +-----+------+
            |                   |                   |
            +-------------+     |     +-------------+
                          |     |     |
                          v     v     v
                    +-----------+-----------+
                    | Stage 3: Anonymized   | (Cross-Evaluation)
                    |          Peer Review  |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    | Stage 4: Dual-Signal  | (Sentence-Transformers
                    |          Detector     |  + DBSCAN Clustering)
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    | Stage 5: Synthesis    | (Groq Llama 3.3 70B
                    |          Combiner     |  or Gemini Flash)
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    | Synthesized Response  |
                    |      with Metrics     |
                    +-----------------------+
```

### 1.2 Target Deployment & Operational Constraints
* **Deployment Target:** Streamlit Community Cloud (Free Tier).
* **RAM Limitation:** Strict **1.0 GB limit**.
* **Memory Footprint Allocation:**
  * `sentence-transformers` library utilizing `all-MiniLM-L6-v2` embedding model: **~400 MB**.
  * Streamlit runtime overhead: **~200 MB**.
  * Remaining allocatable memory for pipeline execution, logs, and database cache: **~400 MB**.
* **Infrastructure Cost:** **$0.00** (runs entirely on free-tier APIs and local CPU embedding calculations; no credit card required).

### 1.3 Key Differentiators vs. Legacy Architectures
* **Mathematical vs. Social Voting:** Instead of relying purely on crowd-sourced model votes (which are vulnerable to systemic bias), VeritasAI blends spatial clustering distances with anonymous, double-blind grading.
* **Aggressive Latency Minimization:** Uses Python’s asynchronous engine (`asyncio.gather`) to fan out API requests. Total execution time is bounded by the slowest individual model rather than the sum of all response latencies.
* **No-Compromise Security on Public Cloud:** Implement database-level encryption (`SQLCipher`) and automatic input sanitization, even when hosted on publicly accessible Streamlit environments.

---

## MODULE 2: Structural Pipeline & Data Modeling

### 2.1 The 5-Stage Execution Pipeline

Every query submitted to VeritasAI must progress sequentially through the following five execution stages:

#### Stage 1: Question Enhancer (`agents/enhancer.py`)
* **Objective:** Standardize, clarify, and optimize raw user input into a highly detailed prompt optimized for multi-LLM comprehension.
* **Input:** Raw query string (or base64 image, or raw transcribed voice).
* **Process:**
  1. Strip leading/trailing whitespaces, reject empty strings, and cap input at 2,000 characters.
  2. Query Groq `llama-3.1-8b-instant` with a specialized system prompt to extract user intent, add critical context, and determine query classification.
* **Query Classifications:** `factual`, `analytical`, `creative`, `code`, or `medical`.
* **Output:** `EnhancedQuery` object containing both the raw text and enhanced prompt.

#### Stage 2: Parallel LLM Dispatcher (`core/dispatcher.py`)
* **Objective:** Execute queries across all active model configurations simultaneously.
* **Input:** `EnhancedQuery` object + optional base64 image payload.
* **Process:**
  1. Iterate over all enabled models defined in `config.yaml`.
  2. Wrap each adapter call in a timeout-safe asynchronous function.
  3. Execute all active adapters in parallel via `asyncio.gather(..., return_exceptions=False)`.
* **Output:** A list of `LLMResult` objects, capturing either successful text generations or gracefully caught failure modes.

#### Stage 3: Anonymized Peer Review (`core/peer_review.py`)
* **Objective:** Perform a double-blind critique where models rate each other's outputs without knowing their origin.
* **Input:** List of successful `LLMResult` objects (requires at least 2 successful responses; otherwise, this stage is skipped and defaults are applied).
* **Process:**
  1. Anonymize successful responses by shuffling their order and mapping them to labels (e.g., `Response A`, `Response B`, `Response C`).
  2. For each successful model, format a single prompt containing all other anonymized responses.
  3. Dispatch these prompts in parallel. Each model evaluates the others based on accuracy, completeness, and clarity, outputting a strictly formatted ranking list: `FINAL RANKING: 1. Response [Letter], 2. Response [Letter]...`.
  4. Parse the output of each peer review using structured regular expressions.
  5. Compute the normalized **Peer Rank Score** ($S_{peer}$) for each model:
$$S_{peer} = 1 - \frac{\text{Average Rank Position}}{n - 1}$$
     *(Where $0$ represents the absolute worst rank position and $n-1$ represents the absolute best position).*
* **Output:** List of `PeerReviewResult` objects.

#### Stage 4: Dual-Signal Detector (`core/detector.py`)
* **Objective:** Compute mathematical outliers and synthesize combined trust metrics.
* **Input:** Successful `LLMResult` objects + `PeerReviewResult` objects.
* **Process:**
  1. Generate text embeddings for all successful responses on a local CPU using `sentence-transformers` (`all-MiniLM-L6-v2`).
  2. Compute a pairwise cosine similarity matrix.
  3. Apply DBSCAN clustering using preconfigured parameters (`eps=0.25`, `min_samples=2`, `metric='cosine'`).
  4. Identify the **Consensus Cluster** as the cluster containing the highest number of members. Any points categorized with a cluster index of `-1` are designated as semantic outliers.
     * *If no valid cluster is identified, all successful models default to a neutral trust score of 0.5.*
  5. Calculate the **Semantic Score** ($S_{semantic}$) for each model as its average cosine similarity to the centroid of the consensus cluster.
  6. Calculate the blended **Trust Score** ($T_{model}$) for each model:
$$T_{model} = (S_{semantic} \times 0.6) + (S_{peer} \times 0.4)$$
  7. Apply **Hallucination Detection Logic**:
     * A model is flagged as hallucinating if and only if both signals agree:
$$\text{is\_hallucinating} = (T_{model} < 0.45) \land (\text{is\_semantic\_outlier} == \text{True})$$
  8. Calculate the overall system **Consensus Ratio** ($R_{consensus}$):
$$R_{consensus} = \frac{\text{Count of Trusted Models}}{\text{Total Successful Models}}$$
* **Output:** A structured `DetectionResult` object.

#### Stage 5: Synthesis Combiner (`agents/combiner.py`)
* **Objective:** Construct the definitive, verified response.
* **Input:** `DetectionResult` object containing only trusted model outputs.
* **Process:**
  * **Low Consensus Branch ($R_{consensus} < 0.5$):** Skip synthesis. Raise the low-consensus flag, display raw model responses side-by-side, and warn the user that a reliable consensus could not be mathematically reached.
  * **High Consensus Branch ($R_{consensus} \ge 0.5$):** Send all trusted responses to the most capable, lowest-latency model available (typically Groq `llama-3.3-70b-versatile` or Gemini `gemini-2.5-flash`). Command the combiner model to merge details, remove redundancies, resolve slight variances, and construct 3 dynamic follow-up questions.
* **Output:** `FinalOutput` dataclass.

---

### 2.2 System Dataclasses

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

class LLMStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class LLMResult:
    model: str
    status: LLMStatus
    response: Optional[str] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    latency_ms: int = 0
    tokens_used: int = 0
    peer_rank_score: float = 0.5  # Default neutral score
    trust_score: float = 0.0
    is_outlier: bool = False

@dataclass
class EnhancedQuery:
    original: str
    enhanced: str
    query_type: str  # factual | analytical | creative | code | medical

@dataclass
class PeerReviewResult:
    model: str
    raw_text: str
    parsed_ranking: List[str]
    peer_rank_score: float

@dataclass
class DetectionResult:
    trusted: List[LLMResult]
    outliers: List[LLMResult]
    trust_scores: Dict[str, float]
    consensus_ratio: float
    low_consensus: bool

@dataclass
class FinalOutput:
    answer: Optional[str]
    consensus_ratio: float
    trust_scores: Dict[str, float]
    peer_rankings: Dict[str, float]
    hallucination_flags: List[str]  # List of flagged model names
    follow_up_questions: List[str]
    low_consensus: bool
    all_results: List[LLMResult]
    total_latency_ms: int
```

---

## MODULE 3: Detailed Functional Requirements & Integrations

### 3.1 Supported LLM Adapters

VeritasAI integrates with multiple free-tier API endpoints. All active integrations are listed below:

| Provider | Target Model | API Endpoint Class | Max Daily Quota | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Google AI Studio** | `gemini-2.5-flash` | `GeminiAdapter` | 1,500 requests | Stage 2, Stage 3, Stage 5 |
| **Groq** | `llama-3.3-70b-versatile` | `GroqAdapter` | 1,000 requests | Stage 2, Stage 3, Stage 5 |
| **Groq** | `llama-3.1-8b-instant` | `GroqAdapter` | 1,000 requests | Stage 1 (Enhancer) |
| **Cerebras** | `llama-3.3-70b` | `CerebrasAdapter` | 1M tokens | Stage 2, Stage 3 |
| **Mistral AI** | `mistral-small-latest` | `MistralAdapter` | 1B tokens/month | Stage 2, Stage 3 |
| **OpenRouter** | `deepseek/deepseek-r1:free` | `OpenRouterAdapter` | Variable (Free Tier) | Stage 2, Stage 3 |
| **NVIDIA NIM** | `deepseek-ai/deepseek-v4-pro`| `NvidiaNimAdapter` | 40 RPM | Stage 2, Stage 3 |
| **Cohere** | `command-r-plus` | `CohereAdapter` | $5 Trial Credit | Stage 2, Stage 3 |

---

### 3.2 Robust Exception Mapping Matrix

All API calls routed through an adapter must execute inside an exception-handling wrapper. This wrapper must catch provider-specific errors and map them to standard internal failure modes:

```python
# Conceptual Error Handling Architecture in Every Adapter
try:
    response = await self.api_call()
except AuthenticationError as e:
    return LLMResult(self.name, LLMStatus.FAILED, error_type="auth", error_msg=str(e))
except RateLimitError as e:
    # Trigger active exponential backoff sleep
    await asyncio.sleep(2 ** attempt)
except asyncio.TimeoutError:
    return LLMResult(self.name, LLMStatus.FAILED, error_type="timeout", error_msg="Exceeded threshold limit")
```

#### Detailed Error Mapping Table:

| Exception Type | Raw Trigger Conditions | System Status Mapping | Immediate System Recovery Behavior |
| :--- | :--- | :--- | :--- |
| **Authentication Error** | 401, 403, invalid api key | `LLMStatus.FAILED` | Skip retries. Immediately flag the model as unconfigured/disabled and notify the system administrator. |
| **Rate Limit Error** | 429, quota exceeded, limit hit | `LLMStatus.FAILED` | Trigger a 2-pass exponential backoff mechanism ($1\text{s} \rightarrow 2\text{s}$). If both retries fail, return a generic failure status. |
| **Timeout Error** | Model fails to respond within 30 seconds | `LLMStatus.FAILED` | Abort the connection task. Do not block the parallel dispatch pipeline. |
| **Network Error** | DNS resolution failure, SSL handshake error | `LLMStatus.FAILED` | Single retry pass after a 2-second sleep. |
| **Server Error** | 500, 502, 503, 504 codes | `LLMStatus.FAILED` | Single retry pass after a 2-second sleep. |
| **Short Response Error**| Successful API response yields $< 20$ words | `LLMStatus.FAILED` | Discard response. Do not forward the output to the Stage 4 detector. |
| **Content Filtered** | API returns a safety block or refusal message | `LLMStatus.FAILED` | Mark response as unsafe. Exclude it from synthesis and downstream processing. |

---

### 3.3 Semantic Session Caching & Rate Tracking

#### Semantic Session Cache
* **Storage Location:** Loaded into `st.session_state` as an active dict mapping.
* **Key Generation:** Calculated as a SHA-256 hash of the sanitized, enhanced query text string.
* **Cached Value:** Fully populated `FinalOutput` dataclass.
* **Execution Bypass:** If a incoming query hash matches an existing key, the entire 5-stage pipeline is bypassed. The cached result is instantly rendered with an explicit `"Cache Hit"` indicator.

#### Rate Limit Tracker
* **Storage Location:** A flat JSON file updated atomically: `logs/rate_tracker.json`.
* **Rate Tracking Key:** Model name + ISO timestamp date string (e.g., `{"groq": {"2026-06-06": 47}}`).
* **Pre-flight Enforcement:** Before triggering any Stage 2 parallel dispatch execution, the system must check the daily quota usage from the tracker against limits defined in `config.yaml`. Any model that has reached its daily limit is skipped.

---

## MODULE 4: Security Architecture & Cryptographic Schemas

### 4.1 Encryption of SQLite Metadata Storage

To ensure the security of queries, logs, and audit metrics, VeritasAI utilizes **SQLCipher** (via the `sqlcipher3` Python driver) to enforce AES-256 transparent encryption on the database file `logs/admin.db`.

```python
import sqlcipher3

def get_secure_db_connection():
    db_key = os.environ.get("DB_ENCRYPTION_KEY")
    conn = sqlcipher3.connect("logs/admin.db")
    conn.execute(f"PRAGMA key='{db_key}'")
    return conn
```

---

### 4.2 SQLite Secure Database Schemas

```sql
-- Schema 1: Every system execution event trace log
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,                  -- INFO | WARN | ERROR | DEBUG
    session_id TEXT NOT NULL,
    event TEXT NOT NULL,                  -- app_start | query_received | l_call_success | etc.
    model TEXT,                           -- Name of target model (if applicable)
    query_hash TEXT,                      -- SHA-256 hash of enhanced query
    latency_ms INTEGER,
    trust_score REAL,
    consensus_ratio REAL,
    error_type TEXT,
    error_msg TEXT,
    tokens_used INTEGER,
    hallucination_flaged INTEGER DEFAULT 0, -- Boolean: 0 | 1
    cache_hit INTEGER DEFAULT 0,            -- Boolean: 0 | 1
    component TEXT                        -- enhancer | dispatcher | detector | combiner | ui
);

-- Schema 2: Persistent records of submitted user queries
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    original_query TEXT NOT NULL,
    enhanced_query TEXT NOT NULL,
    query_type TEXT,
    consensus_ratio REAL,
    final_answer TEXT,
    total_latency_ms INTEGER,
    models_used TEXT,                     -- JSON serialized array of strings
    models_trusted TEXT,                  -- JSON serialized array of strings
    models_flagged TEXT                   -- JSON serialized array of strings
);

-- Schema 3: Individual model evaluations per query
CREATE TABLE model_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    response TEXT,
    trust_score REAL,
    peer_score REAL,
    is_outlier INTEGER DEFAULT 0,
    latency_ms INTEGER,
    tokens_used INTEGER,
    status TEXT,
    error_type TEXT
);

-- Schema 4: User feedback capture
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    vote TEXT NOT NULL,                   -- up | down
    comment TEXT
);
```

---

### 4.3 Data Scrubbing & Privacy Filters

#### SHA-256 Logging Only
Flat-file logs (`logs/veritasai.log`) must never contain raw query text or synthesized responses. The system must only reference queries using their SHA-256 cryptographic hash value. Raw text resides exclusively inside the encrypted `admin.db` file.

#### PII Scrubber Pipeline
Before Stage 1 enhancement, raw user input is processed by a Regex PII scrubber. This utility must identify and redact potential Personally Identifiable Information:
* **Email Scanner:** `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` $\rightarrow$ replaced with `[REDACTED_EMAIL]`.
* **Phone Scanner (US/International):** `\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}` $\rightarrow$ replaced with `[REDACTED_PHONE]`.

#### No-Training Privacy Policy Check
All configurations must use API endpoints where providers explicitly commit to **zero-retention and zero-training policies** on user prompt data.
* *Confirmed Compliant Partners:* Google AI Studio, Groq, Mistral AI, Cerebras, and NVIDIA NIM.

---

### 4.4 Application Access Controls

* **Admin Dashboard Protection:** The administrative metrics interface (`pages/admin.py`) is password-protected.
* **Bcrypt Password Matching:** The input password is hashed and compared using `bcrypt`:

```python
import bcrypt

def verify_admin(input_password):
    stored_hash = st.secrets["admin_password_hash"].encode('utf-8')
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_hash)
```

* **Session Rate Limiting:**
  To prevent denial-of-service attempts on free quotas, rate limits are enforced per session:
  * Maximum queries allowed: **10 queries per hour per unique user session**.
  * Counters are stored in `st.session_state` (`query_count`, `window_start`). If a session exceeds these limits, additional queries are blocked, and an alert is displayed.

---

## MODULE 5: UI/UX & Cognitive Psychology Specifications

### 5.1 Psychological Design & Anxiety Management

The user interface of VeritasAI is designed based on several cognitive psychology principles, focusing on managing user patience during multi-model evaluations.

```
+---------------------------------------------------------------------------------+
|                                 STATE A: LANDING                                |
|                                                                                 |
|                                    VERITAS AI                                   |
|                          "Ask anything. We ask everyone."                       |
|                                                                                 |
|  [                                Search Bar                                 ]  |
|                     ( Text Mode | Voice Mode | Image Mode )                    |
+---------------------------------------------------------------------------------+
```
```
+---------------------------------------------------------------------------------+
|                             STATE B: PROCESSING                                 |
|                                                                                 |
|  Enhancing Query... Done.                                                       |
|                                                                                 |
|  [ Gemini-Flash (S) ] -> [ Cerebras (S) ] -> [ Mistral (S) ] -> [ Groq (L) ]    |
|  ( Spinning loaders transition dynamically into green checkmarks or red crosses) |
|                                                                                 |
|  Evaluating responses in parallel...                                            |
+---------------------------------------------------------------------------------+
```
```
+---------------------------------------------------------------------------------+
|                               STATE C: COMPLETE                                 |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                          Consensus Answer (89%)                           |  |
|  |                                                                           |  |
|  |  Synthesized truth response output text...                                |  |
|  +---------------------------------------------------------------------------+  |
|                                                                                 |
|  +-------------------------------------+  +----------------------------------+  |
|  |          Trust Score Chart          |  |       Individual Responses       |  |
|  |  Gemini:  [|||||||||||||||] 92%     |  |  [+] Gemini Flash Response       |  |
|  |  Cerebras:[|||||||||||||| ] 85%     |  |  [-] Mistral (Outlier Flagged)   |  |
|  |  Mistral: [|||||          ] 30%     |  |  [+] Groq Llama 3 Response       |  |
|  +-------------------------------------+  +----------------------------------+  |
|                                                                                 |
|  * 7 Models Queried  -  5 Consensus  -  1 Flagged  -  Total Latency: 4.8s        |
+---------------------------------------------------------------------------------+
```

---

### 5.2 Key Psychological Principles Applied

#### Hick’s Law (Minimizing Initial Cognitive Load)
* **Design Execution:** The default landing interface (State A) is minimalist. It features only the system logo, the tagline, a single search bar, and 3 mode selection buttons. 
* **Psychological Impact:** Eliminating unnecessary elements on the landing screen helps prevent decision fatigue, guiding users straight to their query entry.

#### Labor Illusion & Managing Waiting Anxiety
* **Design Execution:** When a query is processing, VeritasAI does not display a generic, static loading icon. Instead, it shows a dynamic status row of active model indicators:
  $$\text{[Gemini-Flash]} \rightarrow \text{[Cerebras]} \rightarrow \text{[Mistral]} \rightarrow \text{[Groq]}$$
  As each model finishes executing, its loading spinner changes to a green checkmark or a red cross.
* **Psychological Impact:** Displaying the progress of the underlying processes transforms waiting anxiety into active anticipation. Users perceive the system as working harder and valuing their request, which helps maintain engagement even during longer processing times.

#### Anchoring Effect (Focusing on Synthesized Consensus)
* **Design Execution:** The synthesized, high-confidence answer is displayed first in a large consensus card, complete with a prominent confidence badge (e.g., `Consensus: 89%`). Individual model responses are placed lower on the page.
* **Psychological Impact:** Presenting the verified, aggregated truth first anchors the user's perception with a reliable reference point, before they explore individual model outputs.

#### Progressive Disclosure (Hiding Details Until Requested)
* **Design Execution:** Individual model cards are collapsed by default. Each card displays only the model name, its blended trust score, and a status badge. Users can click to expand the card and view the full raw response.
* **Psychological Impact:** Prevents information overload by allowing users to focus on the main summary, while still giving them the option to dive into technical details on demand.

#### Peak-End Rule (Ending on a Precise, Informative Note)
* **Design Execution:** The bottom of the completed screen features a clean, highly specific system performance summary:
  $$\text{"7 models queried • 5 reached consensus • 1 flagged • 4.8s total execution time"}$$
* **Psychological Impact:** Ending the user's workflow with specific performance metrics reinforces the system's efficiency and transparency, leaving a positive last impression.

---

### 5.3 Color Semantics & Accessibility Guide

To ensure high readability and clear status communication, VeritasAI uses color-coded semantic cues:

* **Green (Hex `#1E4620` to `#2E7D32`):** Representing high trust and consensus. Used for trust scores $\ge 0.70$, successful model checks, and consensus badges.
* **Amber (Hex `#D84315` to `#EF6C00`):** Representing low consensus or potential inconsistencies. Used for trust scores between $0.45$ and $0.69$, as well as the main hallucination warning banner.
* **Red (Hex `#C62828` to `#D32F2F`):** Representing outlier status, failures, or critical errors. Used for trust scores $< 0.45$, rate limit notifications, and failed connection alerts.
* **Minimum Contrast Ratio:** All text-on-background combinations must maintain a contrast ratio of at least **4.5:1** (meeting WCAG AA standards) to support users with color vision deficiencies.

---

## MODULE 6: Implementation Checklist & Production Readiness

### 6.1 Strict Verification Build Order

To ensure a systematic development process, the system must be built in the exact sequence outlined below. Each step must be fully verified before proceeding to the next:

```
+-----------------------------------------------------------------+
|                       PHASE 1: FOUNDATION                       |
|                                                                 |
|  [Step 1] Scaffold directories, config.yaml, & requirements.   |
|  [Step 2] Build dataclasses inside core/result.py.              |
|  [Step 3] Implement secure logging & SQLite tables.             |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                       PHASE 2: ADAPTERS                         |
|                                                                 |
|  [Step 4] Implement LLMAdapter interface base class.            |
|  [Step 5] Write Groq adapter (template for others).             |
|  [Step 6] Build remaining 6 adapters with error wrappers.       |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                        PHASE 3: CORE                            |
|                                                                 |
|  [Step 7] Build rate_tracker.py & sanitizer.py.                 |
|  [Step 8] Write parallel dispatcher with asyncio.               |
|  [Step 9] Implement enhancer.py.                               |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                       PHASE 4: EVALUATION                       |
|                                                                 |
|  [Step 10] Write double-blind peer_review.py logic.             |
|  [Step 11] Write DBSCAN detector.py engine.                     |
|  [Step 12] Implement synthesis combiner.py.                     |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                        PHASE 5: UI & QC                         |
|                                                                 |
|  [Step 13] Implement cache.py mechanism.                        |
|  [Step 14] Build Streamlit UI (States A, B, and C).             |
|  [Step 15] Write admin page and unit testing suite.             |
+-----------------------------------------------------------------+
```

---

### 6.2 Pre-Deployment Verification Checklist

Before deploying VeritasAI to Streamlit Community Cloud, verify that the following checks are complete:

- [ ] **Secret Token Isolation:** Confirm that no API keys are hardcoded in any file. Verify that the `.env` file is added to `.gitignore` and that all production keys are routed through `st.secrets` or environment variables.
- [ ] **SQLCipher Validation:** Verify that calling the admin dashboard with an incorrect encryption key fails to access database entries, confirming active encryption at rest.
- [ ] **PII Scrubber Test:** Pass mock data containing email addresses and phone numbers through the scrubber and verify that all personal identifiers are redacted.
- [ ] **Resource Limit Checks:** Run a mock 5-stage query cycle while monitoring memory usage to ensure peak consumption stays well below the 1.0 GB Streamlit RAM limit.
- [ ] **Adapter Failure Recovery:** Simulate API failures for 3 out of 7 models and verify that the system completes its evaluation, generates consensus statistics, and processes remaining responses without crashing.