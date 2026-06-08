# VeritasAI - Multi-LLM Truth Engine

**Ask anything. We ask everyone.**

VeritasAI is a production-ready Streamlit application that queries multiple AI models simultaneously, detects hallucinations using semantic clustering and peer review, and returns a synthesized, confidence-scored answer.

## Why VeritasAI?

- **Mathematical Hallucination Detection**: DBSCAN clustering on embeddings + cosine similarity to find consensus
- **Dual-Signal Truth Scoring**: 60% mathematical signal + 40% social peer ranking
- **Anonymized Peer Review**: Models critique each other without knowing who wrote what
- **Per-Model Trust Scores**: Visible transparency in which models are reliable
- **7 Free LLM Providers**: Zero cost deployment using free tiers
- **Streamlit Cloud Compatible**: Deploy free on Streamlit Community Cloud

## Architecture

```
Stage 1: Enhancement → Improve raw user query with LLM enhancement
Stage 2: Dispatch → Fire query to 7 LLMs in parallel  
Stage 3: Peer Review → Models anonymously rank each other's responses
Stage 4: Detection → DBSCAN clustering to identify outliers + hallucinations
Stage 5: Synthesis → Combine trusted responses into final answer
```

## Setup

### 1. Clone & Install

```bash
git clone <repo-url>
cd VeritasAI
pip install -r requirements.txt
```

### 2. Create .env File

```bash
cp .env.example .env
```

### 3. Get API Keys (All Free Tier)

| Provider | Signup URL | Free Limit |
|----------|-----------|-----------|
| Groq | https://console.groq.com | 100 calls/day |
| Gemini | https://aistudio.google.com | 50 calls/minute |
| Cerebras | https://cerebras.ai | 300 calls/day |
| Mistral | https://console.mistral.ai | 50 calls/day |
| OpenRouter | https://openrouter.ai | 50 free credits |
| NVIDIA NIM | https://build.nvidia.com/discover/llm | Limited free |
| Cohere | https://dashboard.cohere.ai | 100 calls/month |

### 4. Fill .env with Keys

```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
CEREBRAS_API_KEY=csk-...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
NVIDIA_NIM_API_KEY=nvapi-...
COHERE_API_KEY=...
DB_ENCRYPTION_KEY=your-32-character-random-string
ADMIN_PASSWORD_HASH=bcrypt-hash-of-password
```

### 5. Generate Admin Password Hash

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"
```

### 6. Run Locally

```bash
streamlit run app.py
```

Visit http://localhost:8501

### 7. Deploy to Streamlit Cloud

1. Fork repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New App" → Select your fork
4. Add all `.env` values as secrets in Streamlit dashboard
5. Deploy!

## Key Features

- **Parallel LLM Queries**: 7 models queried simultaneously
- **Hallucination Detection**: DBSCAN clustering on embeddings
- **Peer Review**: Anonymous models rank each other's answers  
- **Dual-Signal Scoring**: Math + social consensus combined
- **Session Cache**: Avoid re-querying same questions
- **PII Scrubbing**: Automatic redaction before API calls
- **Rate Limiting**: 10 queries/hour per session
- **Encrypted Database**: SQLCipher AES-256 at rest
- **Admin Dashboard**: Monitor analytics, logs, model health
- **Voice/Image Input**: Ask with mic or upload images
- **TTS Output**: Listen to synthesized answers

## Project Structure

```
core/
├── adapter.py         # LLMAdapter abstract base
├── result.py          # Data models
├── dispatcher.py      # Parallel dispatch
├── detector.py        # Hallucination detection (DBSCAN)
├── peer_review.py     # Peer ranking
├── enhancer.py        # Query enhancement
├── combiner.py        # Response synthesis
├── logger.py          # Dual logging (file + DB)
├── sanitizer.py       # PII removal
├── rate_tracker.py    # Rate limiting
└── cache.py           # Session caching

agents/
├── groq_adapter.py
├── gemini_adapter.py
├── cerebras_adapter.py
├── mistral_adapter.py
├── openrouter_adapter.py
├── nvidia_nim_adapter.py
├── cohere_adapter.py
├── enhancer.py
└── combiner.py

ui/
├── sidebar.py
├── search_bar.py
├── model_status.py
├── consensus_card.py
├── model_card.py
├── warning_banner.py
├── enhanced_query.py
├── trust_chart.py
├── feedback.py
├── summary_footer.py
└── tts.py

pages/
└── admin.py

tests/
├── test_detector.py
├── test_dispatcher.py
├── test_peer_review.py
└── test_sanitizer.py
```

## Config (config.yaml)

```yaml
app:
  max_input_chars: 2000
  session_rate_limit: 10

pipeline:
  consensus_threshold: 0.5      # warning if below
  trust_threshold: 0.45         # hallucination if below
  semantic_weight: 0.6
  peer_weight: 0.4

detector:
  eps: 0.25                      # DBSCAN distance threshold
  min_samples: 2                 # min models for cluster
  embedding_model: all-MiniLM-L6-v2
```

## Admin Dashboard

Access at `/admin` with password. Shows:
- Total queries, avg confidence, cache hit %, error rate
- Filterable logs table
- Per-model trust score trends
- Model health status

## Security

- API keys in `.env` (gitignored)
- Database encrypted with AES-256
- PII automatically scrubbed
- Session isolation via UUID
- Rate limiting: 10 queries/hour
- Admin password: bcrypt hashed

## License

MIT
