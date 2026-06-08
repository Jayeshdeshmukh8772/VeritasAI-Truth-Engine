"""
VeritasAI — Main Streamlit Application Entry Point.
Orchestrates the 5-stage truth engine pipeline:
  Stage 1: Question Enhancement
  Stage 2: Parallel LLM Dispatch
  Stage 3: Anonymized Peer Review
  Stage 4: DBSCAN Hallucination Detection
  Stage 5: Result Synthesis & Follow-up Generation

Handles async execution via nest_asyncio for Streamlit compatibility.
All state persisted in st.session_state for Streamlit reruns.
"""

import asyncio
import os
import uuid
import yaml
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from typing import Optional

# Streamlit async fix — must be imported before any asyncio.run() calls
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Will use get_event_loop fallback

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="VeritasAI — Multi-LLM Truth Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/veritasai",
        "About": "VeritasAI v1.0 — Ask anything. We ask everyone.",
    },
)

# --- Core infrastructure ---
from core.result import LLMStatus, FinalOutput, EnhancedQuery
from core.sanitizer import InputSanitizer
from core.rate_tracker import RateTracker
from core.logger import VeritasLogger
from core.dispatcher import AsyncDispatcher
from core.peer_review import PeerReviewEngine
from core.detector import HallucinationDetector
from core.cache import ResponseCache
from core.web_search import search_web

# --- Agents ---
from agents.groq_adapter import GroqAdapter
from agents.gemini_adapter import GeminiAdapter
from agents.cerebras_adapter import CerebrasAdapter
from agents.mistral_adapter import MistralAdapter
from agents.nvidia_nim_adapter import NvidiaNimAdapter
from agents.openrouter_adapter import OpenRouterAdapter
from agents.cohere_adapter import CohereAdapter
from agents.enhancer import QuestionEnhancer
from agents.combiner import ResultCombiner

# --- UI Components ---
from ui.sidebar import SidebarComponent
from ui.search_bar import SearchBarComponent
from ui.trust_chart import TrustChartComponent
from ui.feedback import FeedbackComponent
from ui.summary_footer import SummaryFooterComponent

# --- Premium Dark UI styles ---
from styles import (
    inject_styles,
    render_diff,
    render_dispatch_pills,
    render_consensus_card,
    render_model_grid,
)


# ─── Default configuration fallback ─────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "app": {
        "name": "VeritasAI",
        "max_input_chars": 2000,
        "session_rate_limit": 10,
        "min_responses_required": 2,
    },
    "pipeline": {
        "semantic_weight": 0.6,
        "peer_weight": 0.4,
        "consensus_threshold": 0.5,
        "trust_threshold": 0.45,
    },
    "detector": {
        "embedding_model": "all-MiniLM-L6-v2",
        "eps": 0.25,
        "min_samples": 2,
    },
    "features": {
        "cache_enabled": True,
        "voice_input": True,
        "image_input": True,
    },
    "models": [],
}


# ─── System initialization ────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load config.yaml from project root. Falls back to DEFAULT_CONFIG if missing.

    Returns:
        Parsed YAML configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or DEFAULT_CONFIG
        except Exception:
            pass
    return DEFAULT_CONFIG


def initialize_session_state() -> None:
    """
    Bootstrap all session-level objects on first run.
    Skips re-initialization on Streamlit reruns (checks 'initialized' flag).
    """
    if st.session_state.get("initialized"):
        return

    config = load_config()
    st.session_state.config = config
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.query_history = []
    st.session_state.queries_this_hour = 0
    st.session_state.hour_start = datetime.utcnow().hour

    # Core services
    st.session_state.sanitizer = InputSanitizer(
        max_chars=config["app"].get("max_input_chars", 2000)
    )
    st.session_state.rate_tracker = RateTracker()
    st.session_state.logger = VeritasLogger()
    st.session_state.cache = ResponseCache()

    # Build adapter list — only enabled adapters with configured keys
    adapter_pool = [
        GroqAdapter(),
        GeminiAdapter(),
        CerebrasAdapter(),
        MistralAdapter(),
        NvidiaNimAdapter(),
        OpenRouterAdapter(model_id="deepseek/deepseek-r1:free"),
        OpenRouterAdapter(model_id="meta-llama/llama-4-maverick:free"),
        CohereAdapter(),
    ]
    st.session_state.adapters = adapter_pool

    adapter_map = {adapter.name: adapter for adapter in adapter_pool}

    # Pipeline components
    st.session_state.enhancer = QuestionEnhancer(
        groq_adapter=adapter_pool[0],  # Groq is fastest for enhancement
        sanitizer=st.session_state.sanitizer,
    )
    st.session_state.combiner = ResultCombiner(
        fallback_adapters=[adapter_pool[0], adapter_pool[1]]  # Groq, then Gemini
    )
    st.session_state.detector = HallucinationDetector(
        model_name=config["detector"].get("embedding_model", "all-MiniLM-L6-v2"),
        eps=config["detector"].get("eps", 0.25),
        min_samples=config["detector"].get("min_samples", 2),
    )
    st.session_state.peer_engine = PeerReviewEngine(adapter_map)
    st.session_state.dispatcher = AsyncDispatcher(
        adapter_pool, st.session_state.rate_tracker, config
    )

    st.session_state.model_health = {}
    st.session_state.initialized = True

    st.session_state.logger.log_event(
        "INFO", st.session_state.session_id,
        "app_start", "app",
        message="VeritasAI session initialized",
    )


# ─── Pipeline execution ───────────────────────────────────────────────────────

def run_pipeline(raw_query: str, image_b64: Optional[str] = None) -> FinalOutput:
    """
    Execute the full 5-stage pipeline synchronously (wraps async pipeline).

    Args:
        raw_query: Raw user input string
        image_b64: Optional base64-encoded image for multimodal queries

    Returns:
        FinalOutput with all results, scores, and synthesized answer
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(_pipeline_async(raw_query, image_b64))
    # _pipeline_async returns a FinalOutput directly (enhanced_query stored in session_state)
    return result


async def _pipeline_async(raw_query: str, image_b64: Optional[str] = None):
    """
    Full 5-stage async pipeline execution.

    Args:
        raw_query: Raw user input
        image_b64: Optional base64 image

    Returns:
        Tuple of (FinalOutput, EnhancedQuery) with all pipeline results
    """
    state = st.session_state
    log = state.logger
    sess_id = state.session_id
    config = state.config
    start_time = datetime.utcnow()

    sem_weight = st.session_state.get("sem_weight", config["pipeline"].get("semantic_weight", 0.6))
    peer_weight = round(1.0 - sem_weight, 2)

    log.log_event("INFO", sess_id, "query_received", "app",
                  message=f"Query received ({len(raw_query)} chars)")

    # ── STAGE 1: Question Enhancement ──────────────────────────────────────────
    if st.session_state.get("feat_enhancer", True):
        enhanced_obj: EnhancedQuery = await state.enhancer.enhance(raw_query)
    else:
        from core.result import EnhancedQuery
        cleaned = state.sanitizer.sanitize(raw_query)
        enhanced_obj = EnhancedQuery(
            original=cleaned, enhanced=cleaned,
            query_type=state.sanitizer.detect_query_type(cleaned)
        )

    log.log_event("INFO", sess_id, "query_enhanced", "enhancer",
                  message=f"Query type: {enhanced_obj.query_type}")

    # ── Cache lookup ───────────────────────────────────────────────────────────
    q_hash = state.cache.compute_signature(enhanced_obj.enhanced)
    if st.session_state.get("feat_cache", True):
        cached = state.cache.lookup(enhanced_obj.enhanced)
        if cached:
            log.log_event("INFO", sess_id, "cache_hit", "cache",
                          query_hash=q_hash, cache_hit=1)
            return cached

    log.log_event("INFO", sess_id, "cache_miss", "cache", query_hash=q_hash)

    # Store the enhanced query for later display
    st.session_state["current_enhanced"] = enhanced_obj

    # ── STAGE 2: Parallel LLM Dispatch (with Live Web Search / RAG) ───────────
    dispatch_prompt = enhanced_obj.enhanced
    
    if st.session_state.get("feat_search", True):
        with st.spinner("🌐 Querying live web for real-time facts..."):
            search_results = search_web(raw_query)
            if search_results:
                log.log_event("INFO", sess_id, "web_search_success", "web_search",
                              message=f"Retrieved real-time search context for query: {raw_query[:50]}...")
                dispatch_prompt = (
                    f"{enhanced_obj.enhanced}\n\n"
                    f"[REAL-TIME GROUNDING CONTEXT (WEB SEARCH)]:\n"
                    f"{search_results}\n\n"
                    f"Instructions: Use the real-time grounding context above to provide an accurate, up-to-date response. "
                    f"Prioritize the search context facts over your historical training knowledge cutoff."
                )

    fast_mode = st.session_state.get("fast_mode", False)
    active_adapters = state.adapters[:3] if fast_mode else state.adapters

    # Initialize model statuses for dynamic display
    model_statuses = {}
    model_cfg_map = {cfg['name']: cfg for cfg in config.get('models', [])}
    for adapter in active_adapters:
        slug = state.dispatcher._get_slug(adapter.name)
        cfg = model_cfg_map.get(slug, {"enabled": True, "daily_limit": 100})
        
        if not cfg.get("enabled", True):
            model_statuses[adapter.name] = {"status": "skipped", "latency_ms": None}
        elif not state.rate_tracker.check_quota(slug, cfg.get("daily_limit", 100)):
            model_statuses[adapter.name] = {"status": "skipped", "latency_ms": None}
        else:
            model_statuses[adapter.name] = {"status": "loading", "latency_ms": None}

    # Map query classifications to psychological wait anxiety loading statements
    q_type = enhanced_obj.query_type.lower()
    if q_type == "medical":
        loading_msg = "⚕️ Consulting clinical database guidelines and cross-referencing health registries..."
    elif q_type == "code":
        loading_msg = "💻 Parsing code syntax trees, checking import safety, and verifying logical structures..."
    elif q_type == "creative":
        loading_msg = "🎨 Priming style registers, gathering creative ideas, and shifting variation paths..."
    elif q_type == "analytical":
        loading_msg = "🧩 Decomposing variables, analyzing logical premises, and verifying argument consistency..."
    else:  # factual
        loading_msg = "🌐 Querying global information channels and cross-referencing real-time search context..."

    # Render initial dynamic loading row
    status_placeholder = st.empty()
    with status_placeholder.container():
        st.info(loading_msg)
        st.markdown(render_dispatch_pills(model_statuses), unsafe_allow_html=True)

    # Wrap adapter call in an async task that updates the status row upon completion
    async def run_and_update_task(adapter, prompt, image_b64, slug):
        if model_statuses[adapter.name]["status"] == "skipped":
            return await state.dispatcher._build_skipped(adapter.name, "disabled")
            
        result = await state.dispatcher._execute_and_track(adapter, prompt, image_b64, slug)
        
        status_str = result.status.value  # "success" | "failed" | "skipped"
        model_statuses[adapter.name] = {
            "status": status_str,
            "latency_ms": result.latency_ms if status_str == "success" else None
        }
        
        # Rerender dynamic pills
        with status_placeholder.container():
            st.info(loading_msg)
            st.markdown(render_dispatch_pills(model_statuses), unsafe_allow_html=True)
            
        return result

    # Execute all adapter calls concurrently
    tasks = []
    for adapter in active_adapters:
        slug = state.dispatcher._get_slug(adapter.name)
        tasks.append(run_and_update_task(adapter, dispatch_prompt, image_b64, slug))

    raw_results = await asyncio.gather(*tasks)
    status_placeholder.empty()

    successful = [r for r in raw_results if r.status == LLMStatus.SUCCESS]

    # Update model health
    health = {}
    for r in raw_results:
        health[r.model] = r.status == LLMStatus.SUCCESS
    st.session_state.model_health = health

    log.log_event("INFO", sess_id, "dispatch_complete", "dispatcher",
                  query_hash=q_hash,
                  message=f"{len(successful)}/{len(raw_results)} models succeeded")

    # ── Minimum threshold check ────────────────────────────────────────────────
    min_required = config["app"].get("min_responses_required", 2)
    if len(successful) == 0:
        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        return FinalOutput(
            answer="⚠️ All models are currently unavailable. Please try again in a few minutes.",
            consensus_ratio=0.0, trust_scores={}, peer_rankings={},
            hallucination_flags=[], follow_up_questions=[],
            low_consensus=True, all_results=raw_results, total_latency_ms=latency,
        )

    if len(successful) < min_required:
        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        # Fall back to best single response
        best = successful[0] if successful else None
        return FinalOutput(
            answer=best.response if best else "Insufficient model responses.",
            consensus_ratio=0.5, trust_scores={r.model: 0.5 for r in successful},
            peer_rankings={}, hallucination_flags=[], follow_up_questions=[],
            low_consensus=True, all_results=raw_results, total_latency_ms=latency,
        )

    # ── STAGE 3: Anonymized Peer Review ───────────────────────────────────────
    use_peer = st.session_state.get("feat_peer", True) and not fast_mode
    if use_peer and len(successful) >= 2:
        reviewed_results = await state.peer_engine.run_review(enhanced_obj.enhanced, successful)
        log.log_event("INFO", sess_id, "peer_review_complete", "peer_review",
                      query_hash=q_hash, message=f"Peer review done for {len(reviewed_results)} models")
    else:
        reviewed_results = successful  # peer_rank_score stays at default 0.5

    # ── STAGE 4: Hallucination Detection ──────────────────────────────────────
    detection = state.detector.analyze(
        reviewed_results,
        semantic_weight=sem_weight,
        peer_weight=peer_weight,
    )

    log.log_event(
        "INFO", sess_id, "consensus_computed", "detector",
        query_hash=q_hash, consensus_ratio=detection.consensus_ratio,
        message=f"Consensus ratio: {detection.consensus_ratio:.2f}, outliers: {len(detection.outliers)}",
    )

    for outlier in detection.outliers:
        log.log_event("WARN", sess_id, "hallucination_flagged", "detector",
                      model=outlier.model, query_hash=q_hash,
                      trust_score=outlier.trust_score, hallucination_flagged=1)

    # ── STAGE 5: Synthesis ─────────────────────────────────────────────────────
    flagged_models = [r.model for r in detection.outliers]

    if detection.high_dissent:
        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        # Format a side-by-side comparison of the divergent model answers into answer_text
        successful_reviewed = [r for r in reviewed_results if r.status == LLMStatus.SUCCESS]
        if len(successful_reviewed) >= 2:
            model_a, model_b = successful_reviewed[0], successful_reviewed[1]
            answer_text = (
                f"### ⚠️ High Dissent / Contradictory Scenario\n"
                f"The active models returned highly contradictory viewpoints (semantic similarity < 0.50). "
                f"A consensus could not be safely reached.\n\n"
                f"#### Viewpoint A: {model_a.model}\n"
                f"{model_a.response}\n\n"
                f"--- \n\n"
                f"#### Viewpoint B: {model_b.model}\n"
                f"{model_b.response}"
            )
        else:
            answer_text = "⚠️ High dissent detected, but insufficient model responses are available to display details."

        output = FinalOutput(
            answer=answer_text,
            consensus_ratio=detection.consensus_ratio,
            trust_scores=detection.trust_scores,
            peer_rankings={r.model: r.peer_rank_score for r in reviewed_results},
            hallucination_flags=flagged_models,
            follow_up_questions=[],
            low_consensus=True,
            all_results=raw_results,
            total_latency_ms=latency,
            high_dissent=True,
        )
    elif detection.low_consensus:
        # Low consensus: skip synthesis, show warning
        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        output = FinalOutput(
            answer=None,
            consensus_ratio=detection.consensus_ratio,
            trust_scores=detection.trust_scores,
            peer_rankings={r.model: r.peer_rank_score for r in reviewed_results},
            hallucination_flags=flagged_models,
            follow_up_questions=[],
            low_consensus=True,
            all_results=raw_results,
            total_latency_ms=latency,
            high_dissent=False,
        )
    else:
        final_answer, follow_ups = await state.combiner.synthesize_with_followups(
            enhanced_obj.enhanced, detection.trusted
        )
        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        log.log_event("INFO", sess_id, "synthesis_complete", "combiner",
                      query_hash=q_hash, latency_ms=latency,
                      consensus_ratio=detection.consensus_ratio)

        output = FinalOutput(
            answer=final_answer,
            consensus_ratio=detection.consensus_ratio,
            trust_scores=detection.trust_scores,
            peer_rankings={r.model: r.peer_rank_score for r in reviewed_results},
            hallucination_flags=flagged_models,
            follow_up_questions=follow_ups,
            low_consensus=False,
            all_results=raw_results,
            total_latency_ms=latency,
            high_dissent=False,
        )

    # Log complete query record
    log.log_query(
        session_id=sess_id,
        query_hash=q_hash,
        original_query=enhanced_obj.original,
        enhanced_query=enhanced_obj.enhanced,
        query_type=enhanced_obj.query_type,
        consensus_ratio=detection.consensus_ratio,
        final_answer=output.answer,
        total_latency_ms=output.total_latency_ms,
        models_used=[r.model for r in raw_results],
        models_trusted=[r.model for r in detection.trusted],
        models_flagged=flagged_models,
    )

    # Log per-model responses
    for r in raw_results:
        log.log_model_response(
            query_hash=q_hash,
            model=r.model,
            response=r.response,
            trust_score=r.trust_score,
            peer_score=r.peer_rank_score,
            is_outlier=r.is_outlier,
            latency_ms=r.latency_ms,
            tokens_used=r.tokens_used,
            status=r.status.value,
            error_type=r.error_type,
        )

    log.log_event("INFO", sess_id, "query_complete", "pipeline",
                  query_hash=q_hash, latency_ms=output.total_latency_ms,
                  consensus_ratio=detection.consensus_ratio)

    # Cache the result
    if st.session_state.get("feat_cache", True):
        state.cache.set(enhanced_obj.enhanced, output)

    return output


# ─── Session rate limiting ────────────────────────────────────────────────────

def check_rate_limit() -> bool:
    """
    Check if this session has exceeded the hourly query limit.

    Returns:
        True if under limit, False if limit exceeded
    """
    current_hour = datetime.utcnow().hour
    if st.session_state.get("hour_start") != current_hour:
        st.session_state.queries_this_hour = 0
        st.session_state.hour_start = current_hour

    limit = st.session_state.config["app"].get("session_rate_limit", 10)
    return st.session_state.get("queries_this_hour", 0) < limit


# ─── Main UI ──────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main application entry point.
    Manages 3 UI states: LANDING → RUNNING → RESULTS.
    """
    # Initialize all session state on first run
    initialize_session_state()

    # Inject the premium dark styles
    inject_styles()

    # Render sidebar (always visible)
    SidebarComponent.render_controls()

    # Check for follow-up or history reload triggers
    pending_query: Optional[str] = None
    if "followup_query" in st.session_state:
        pending_query = st.session_state.pop("followup_query")
    elif "reload_query" in st.session_state:
        pending_query = st.session_state.pop("reload_query")

    # ── STATE A: LANDING ──────────────────────────────────────────────────────
    if "current_output" not in st.session_state and not pending_query:
        query_input, image_b64 = SearchBarComponent.render()

        if query_input:
            _handle_query(query_input, image_b64)

    # ── STATE B/C: RESULTS DISPLAYED ─────────────────────────────────────────
    else:
        # Allow new query at the top
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                new_query = st.text_input(
                    "New query",
                    placeholder="Ask a new question...",
                    label_visibility="collapsed",
                    key="quick_query_bar",
                )
            with col2:
                new_search_btn = st.button("🔍 Search", use_container_width=True)

        if new_search_btn and new_query and new_query.strip():
            pending_query = new_query.strip()

        # Handle pending query (follow-up, history, or new search)
        if pending_query:
            del st.session_state["current_output"]
            if "current_enhanced" in st.session_state:
                del st.session_state["current_enhanced"]
            _handle_query(pending_query, None)
            return

        # Display results
        output: FinalOutput = st.session_state.get("current_output")
        enhanced: EnhancedQuery = st.session_state.get("current_enhanced")

        if output and enhanced:
            _render_results(output, enhanced)


def _handle_query(raw_query: str, image_b64: Optional[str]) -> None:
    """
    Execute the pipeline for a new query and display results.

    Args:
        raw_query: The user's query string
        image_b64: Optional base64 image
    """
    # Rate limit check
    if not check_rate_limit():
        st.error("⚠️ You've reached the hourly query limit (10 queries/hour). Please wait before submitting again.")
        return

    # Add to history
    SidebarComponent.add_to_history(raw_query)
    st.session_state.queries_this_hour = st.session_state.get("queries_this_hour", 0) + 1

    # ── STATE B: RUNNING ─────────────────────────────────────────────────────
    with st.spinner("🧠 VeritasAI is querying the council..."):
        status_placeholder = st.empty()
        with status_placeholder.container():
            st.info(
                f"🚀 Dispatching your query to **{len(st.session_state.adapters)}** AI models simultaneously. "
                "Running hallucination detection and peer review..."
            )

        try:
            output = run_pipeline(raw_query, image_b64)
            st.session_state["current_output"] = output
            # current_enhanced is set inside _pipeline_async via session_state

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            st.session_state.logger.log_event(
                "ERROR", st.session_state.session_id,
                "pipeline_error", "app",
                error_type=type(e).__name__,
                error_msg=str(e)[:200],
                message="Unhandled pipeline exception",
            )
            return

    # Clear status and trigger rerun to show results
    status_placeholder.empty()
    st.rerun()


def _render_results(output: FinalOutput, enhanced: Optional[EnhancedQuery]) -> None:
    """
    Render the complete results page (STATE C) using the premium dark UI style elements.
    """
    # 1. Operational Dispatch Tracker (Status Pills)
    st.markdown("### Operational Dispatch Tracker")
    model_statuses = {}
    for res in output.all_results:
        status_str = res.status.value  # "success" | "failed" | "skipped"
        model_statuses[res.model] = {
            "status": status_str,
            "latency_ms": res.latency_ms if status_str == "success" else None
        }
    st.markdown(render_dispatch_pills(model_statuses), unsafe_allow_html=True)

    # 2. Enhanced query diff (Query Optimization Card)
    if enhanced and enhanced.original != enhanced.enhanced:
        st.markdown("### Query Optimization")
        st.markdown(render_diff(enhanced.original, enhanced.enhanced, enhanced.query_type), unsafe_allow_html=True)

    # 3. Consensus Answer (Hero Card)
    st.markdown("### Consensus Answer")
    models_used = sum(1 for r in output.all_results if r.status == LLMStatus.SUCCESS)
    models_total = len(output.all_results)
    
    st.markdown(
        render_consensus_card(
            answer=output.answer if output.answer else "No synthesis output generated due to low model consensus.",
            confidence_pct=int(output.consensus_ratio * 100),
            models_used=models_used,
            models_total=models_total,
            follow_ups=None,  # Handled below as interactive Streamlit buttons
            low_consensus=output.low_consensus,
            high_dissent=output.high_dissent,
        ),
        unsafe_allow_html=True,
    )

    # Copy and read aloud buttons row
    if output.answer:
        action_col1, action_col2, _ = st.columns([2, 2, 8])
        with action_col1:
            from ui.tts import TTSComponent
            TTSComponent.render_button(output.answer, "🔊 Read Aloud")
        with action_col2:
            import base64
            b64_answer = base64.b64encode(output.answer.encode('utf-8')).decode('utf-8')
            button_html = (
                f'<button onclick="navigator.clipboard.writeText(decodeURIComponent(escape(atob(\'{b64_answer}\')))).then(() => {{ '
                f'this.textContent=\'✅ Copied!\'; setTimeout(() => this.textContent=\'📋 Copy Answer\', 1500); '
                f'}})" style="background: transparent; border: 0.5px solid #1e2535; border-radius: 7px; '
                f'color: #64748b; font-family: \'Outfit\', sans-serif; font-size: 12px; padding: 7px 14px; '
                f'cursor: pointer; transition: all 0.15s ease; width: 100%; height: 38px;" '
                f'onmouseover="this.style.borderColor=\'#2d3561\'; this.style.color=\'#a5b4fc\'; this.style.background=\'#13182a\';" '
                f'onmouseout="this.style.borderColor=\'#1e2535\'; this.style.color=\'#64748b\'; this.style.background=\'transparent\';">'
                f'📋 Copy Answer</button>'
            )
            st.markdown(button_html, unsafe_allow_html=True)

    # 4. Interactive Follow-up Questions (Native Streamlit Buttons with theme overrides)
    if output.follow_up_questions and not output.high_dissent:
        st.write("")
        st.markdown("**💡 Follow-up Questions:**")
        cols = st.columns(len(output.follow_up_questions[:3]))
        for i, q in enumerate(output.follow_up_questions[:3]):
            with cols[i]:
                if st.button(f"→ {q}", key=f"followup_{i}", use_container_width=True):
                    st.session_state["followup_query"] = q
                    st.rerun()

    # 5. Trust Score Chart (Native Plotly Chart inheriting dark dashboard style overrides)
    st.write("")
    st.markdown("### Council Confidence Grid")
    TrustChartComponent.render(output)

    # 6. Responsive Individual Model Cards
    st.markdown("### Individual Model Responses")
    grid_results = []
    for res in output.all_results:
        grid_results.append({
            "name": res.model,
            "response": res.response if res.response else "",
            "trust_score": res.trust_score,
            "latency_ms": res.latency_ms,
            "peer_rank_score": res.peer_rank_score if res.status == LLMStatus.SUCCESS else None,
            "is_outlier": res.is_outlier,
            "status": res.status.value,
        })
    st.markdown(render_model_grid(grid_results), unsafe_allow_html=True)

    # 7. Feedback Component
    q_hash = st.session_state.cache.compute_signature(
        enhanced.enhanced if enhanced else "query"
    )
    FeedbackComponent.render(
        query_hash=q_hash,
        session_id=st.session_state.session_id,
        logger=st.session_state.logger,
    )

    # 8. Summary Footer
    SummaryFooterComponent.render(output)


# ─── Entry point ──────────────────────────────────────────────────────────────

main()