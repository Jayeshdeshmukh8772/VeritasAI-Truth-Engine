# VeritasAI Implementation Completion Guide

## Current Status

✅ **Completed Foundation**:
- All core data models and interfaces
- Logging, caching, rate tracking, sanitization
- Async dispatcher
- Groq adapter (template for others)
- Configuration files
- Project documentation

⏳ **To Complete**: ~30% remaining
- 6 remaining LLM adapters (follow Groq pattern)
- 4 core algorithm files (detector, peer_review, enhancer, combiner_logic)
- 11 UI components
- Main app.py orchestrator
- Admin dashboard
- Tests

## Quick Start to Complete

### Phase 1: Complete the 6 Remaining Adapters (1-2 hours)

Copy `agents/groq_adapter.py` as template and modify for each provider:

**agents/gemini_adapter.py**
```python
# Key changes from Groq template:
import google.generativeai as genai
# Set: supports_images = True
# Handle base64 image in content parts
# Use model: "gemini-2.5-flash"
```

**agents/cerebras_adapter.py**
```python
from cerebras.cloud.sdk import AsyncCerebras
# Model: "llama-3.3-70b"
# Timeout: 20s, max_retries: 1
```

**agents/mistral_adapter.py**
```python
from mistralai import AsyncMistral
# Model: "mistral-small-latest"
# Timeout: 40s, max_retries: 1
```

**agents/openrouter_adapter.py**
```python
from openai import AsyncOpenAI
# Base URL: https://openrouter.ai/api/v1
# Try models: [deepseek/deepseek-r1:free, meta-llama/llama-4-maverick:free]
# Timeout: 45s
```

**agents/nvidia_nim_adapter.py**
```python
from openai import AsyncOpenAI
# Base URL: https://integrate.api.nvidia.com/v1
# Model: "deepseek-ai/deepseek-v4-pro"
# Timeout: 30s
```

**agents/cohere_adapter.py**
```python
from cohere import AsyncClient
# Model: "command-r-plus"
# Single attempt
```

All adapters must:
- Inherit from `LLMAdapter`
- Never raise exceptions
- Return `LLMResult` with proper status
- Handle: auth errors, rate limits, timeouts, server errors
- Filter responses < 20 words as "empty_response"

### Phase 2: Core Algorithm Files (2-3 hours)

**core/detector.py** - Hallucination Detection
```python
class HallucinationDetector:
    def detect(self, results: List[LLMResult]) -> DetectionResult:
        # 1. Get embeddings (sentence-transformers)
        # 2. Run DBSCAN clustering  
        # 3. Find largest cluster = consensus
        # 4. Score models: 1.0 if in cluster, 0.0 if outlier
        # 5. Return DetectionResult with scores + consensus_ratio
```

**core/peer_review.py** - Peer Ranking
```python
class PeerReviewEngine:
    async def review(self, results: List[LLMResult]) -> Dict[str, float]:
        # 1. For each model, create ranking prompt
        # 2. Model ranks all responses (A, B, C, ...)
        # 3. Parse ranking text to extract order
        # 4. Compute score: higher rank = higher score
        # 5. Return dict: {model_name: score}
```

**core/combiner_logic.py** - Response Synthesis
```python
class ResponseCombiner:
    async def synthesize(self, trusted_results: List[LLMResult]) -> tuple:
        # 1. Pick fastest available model (Groq > Gemini > any)
        # 2. Create synthesis prompt combining responses
        # 3. Generate final answer
        # 4. Generate 3 follow-up questions
        # 5. Return (final_answer, follow_up_questions)
```

**core/enhancer.py** - Query Enhancement
```python
class QuestionEnhancer:
    async def enhance(self, query: str) -> EnhancedQuery:
        # 1. Use Groq llama-3.1-8b-instant to rewrite query
        # 2. Detect query type (factual/analytical/creative/code/medical)
        # 3. Return EnhancedQuery(original, enhanced, type)
```

### Phase 3: UI Components (2-3 hours)

Create `ui/*.py` files with Streamlit components:

```python
# Template pattern for all UI components
import streamlit as st
from core.result import FinalOutput, LLMResult

class ComponentName:
    @staticmethod
    def render(data):
        """Render component using st.* functions."""
        # Use st.container(), st.columns(), st.metric(), etc.
        # Layout responsively
        # Handle None/empty states
```

**Key Components**:
- `sidebar.py`: Model toggle switches + settings
- `search_bar.py`: Text input (st.text_area) + voice + image buttons
- `model_status.py`: Live grid of model status pills
- `consensus_card.py`: Main answer card with confidence %
- `model_card.py`: Expandable cards per model response
- `warning_banner.py`: Amber banner if low consensus or hallucination
- `trust_chart.py`: Line chart of trust scores (plotly/matplotlib)
- `enhanced_query.py`: Side-by-side original vs enhanced
- `feedback.py`: Thumbs up/down + text comment
- `summary_footer.py`: Results summary at bottom
- `tts.py`: Listen button with audio playback

### Phase 4: Main Application Files (2-3 hours)

**app.py** - Main Orchestrator
```python
import streamlit as st
import asyncio
from core.result import FinalOutput
from core.dispatcher import AsyncDispatcher
from core.detector import HallucinationDetector
# ... other imports

def main():
    st.set_page_config(page_title="VeritasAI", layout="wide")
    
    # 1. Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.logger = VeritasLogger()
        st.session_state.dispatcher = AsyncDispatcher(adapters, rate_tracker, config)
        st.session_state.cache = {}
    
    # 2. Render UI components
    # - Sidebar
    # - Search bar
    
    # 3. Handle query submission
    if user_query:
        # - Enhance query
        # - Dispatch to all models  
        # - Detect hallucinations
        # - Combine responses
        # - Display results
        
        output: FinalOutput = asyncio.run(execute_pipeline(user_query))
        
        # Render results using UI components

if __name__ == "__main__":
    main()
```

**pages/admin.py** - Admin Dashboard
```python
import streamlit as st
import bcrypt
from core.logger import VeritasLogger
# ... imports

def admin_page():
    st.set_page_config(page_title="Admin - VeritasAI", layout="wide")
    
    # 1. Password gate
    password = st.text_input("Admin Password", type="password")
    password_hash = os.getenv("ADMIN_PASSWORD_HASH", "").encode()
    if not bcrypt.checkpw(password.encode(), password_hash):
        st.error("Invalid password")
        return
    
    # 2. Metric cards (4 columns)
    # - Total queries
    # - Avg confidence %
    # - Cache hit rate %
    # - Model error rate %
    
    # 3. Logs table (filterable)
    # Query logs from SQLite database
    
    # 4. Per-model trust score line chart
    
    # 5. Model health table

if __name__ == "__main__":
    admin_page()
```

### Phase 5: Tests (1-2 hours)

**tests/test_detector.py**
```python
import pytest
from core.detector import HallucinationDetector
from core.result import LLMResult, LLMStatus

def test_detector_clusters_similar_responses():
    # Create 3 similar responses + 1 outlier
    # Run detector
    # Assert: outlier flagged, 3 others in consensus
```

**tests/test_dispatcher.py**
```python
@pytest.mark.asyncio
async def test_dispatcher_runs_parallel():
    # Create mock adapters
    # Run dispatch_all()
    # Assert: all adapters called in parallel
    # Assert: results contain all model responses
```

**tests/test_peer_review.py**
```python
def test_peer_review_parsing():
    # Create mock responses
    # Run peer review
    # Assert: rankings parsed correctly
    # Assert: scores are 0.0-1.0
```

**tests/test_sanitizer.py**
```python
def test_sanitizer_removes_email():
    text = "Contact me at test@example.com"
    sanitized, redacted = InputSanitizer.sanitize(text)
    assert "[EMAIL]" in sanitized
    assert "test@example.com" in redacted
```

## Testing Your Implementation

```bash
# 1. Test imports
python -c "from core import *; from agents import *"

# 2. Run locally
streamlit run app.py

# 3. Run tests
pytest tests/ -v

# 4. Type check
mypy . --ignore-missing-imports

# 5. Test adapters individually
# Create test_adapters.py and test each adapter's call() method
```

## Deployment Checklist

- [ ] All 7 adapters implemented + tested
- [ ] All core algorithms working + tested
- [ ] All UI components render + interactive
- [ ] app.py runs locally without errors
- [ ] Admin page password works
- [ ] Cache stores/retrieves results
- [ ] Rate tracking works
- [ ] Logging to file + DB works
- [ ] README has setup instructions
- [ ] All dependencies in requirements.txt
- [ ] No API keys in source code
- [ ] Test suite passes
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud + test

## Common Issues & Solutions

**Import Error**: Ensure all files created in correct directories
**Type Errors**: Use `Optional[str]` not `str | None` (Python 3.11+)
**Async Issues**: Use `asyncio.run()` in Streamlit context
**API Key Errors**: Check `.env` file has all 7 keys
**GPU Memory**: Sentence-transformers uses ~400MB (should be fine on 1GB Streamlit tier)

## Next Steps

1. Copy Groq adapter and modify for 6 other providers
2. Implement 4 core algorithm files from sketches above
3. Create 11 UI components using Streamlit templates
4. Implement app.py and admin.py
5. Write tests
6. Run locally + test
7. Deploy to Streamlit Cloud

## Questions?

Refer to:
- `config.yaml` for all model configs
- `core/result.py` for data model shapes
- `core/adapter.py` for interface contract
- Existing `agents/groq_adapter.py` as implementation template
- Streamlit docs: https://docs.streamlit.io

Good luck! 🚀
