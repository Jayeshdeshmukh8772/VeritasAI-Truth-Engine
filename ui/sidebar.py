"""
VeritasAI Sidebar Control Panel — Glassmorphism Edition.
Frosted glass panel with live activity monitor, pipeline controls, model health,
usage tracker, and query history. No external links (GitHub/Admin/Docs removed).
"""

import os
import streamlit as st
from typing import List


class SidebarComponent:
    """Renders the full glassmorphic sidebar with all controls and query history."""

    @staticmethod
    def render_controls() -> None:
        with st.sidebar:

            # ── Branding ──────────────────────────────────────────────────────
            st.markdown("""
<div class="sb-brand">
  <div class="sb-logo-ring">⚖️</div>
  <div>
    <div class="sb-brand-name">VeritasAI</div>
    <div class="sb-brand-sub">Multi-LLM Truth Engine</div>
  </div>
  <div class="sb-live-badge"><span class="sb-live-dot"></span>Live</div>
</div>
""", unsafe_allow_html=True)

            # ── Activity Monitor ──────────────────────────────────────────────
            queries_today = st.session_state.get("queries_this_hour", 0)
            model_health = st.session_state.get("model_health", {})
            healthy_count = sum(1 for v in model_health.values() if v)
            total_models = len(model_health) if model_health else 8

            st.markdown(f"""
<div class="sb-section-label">Activity Monitor</div>
<div class="sb-glass-card">
  <div class="sb-activity-grid">
    <div class="sb-activity-cell">
      <div class="sb-activity-value">{queries_today}</div>
      <div class="sb-activity-desc">Queries this hour</div>
    </div>
    <div class="sb-activity-cell">
      <div class="sb-activity-value" style="color:#4ade80">{healthy_count}<span style="font-size:12px;color:#64748b">/{total_models}</span></div>
      <div class="sb-activity-desc">Models online</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Control Panel ─────────────────────────────────────────────────
            st.markdown('<div class="sb-section-label">Control Panel</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-glass-card sb-controls-card">', unsafe_allow_html=True)

            fast_mode = st.toggle(
                "⚡ Fast Mode",
                value=st.session_state.get("fast_mode", False),
                key="fast_mode",
                help="3 models only — no peer review. Faster but less thorough.",
            )
            st.toggle("🌐 Live Web Search", value=True, key="feat_search",
                      help="Real-time grounding context via DuckDuckGo")
            st.toggle("✨ Query Enhancement", value=True, key="feat_enhancer",
                      help="Rewrites query for maximum clarity before dispatch")
            st.toggle("💾 Response Cache", value=True, key="feat_cache",
                      help="Cache identical queries to save API calls")
            st.toggle("🔬 Peer Review", value=True, key="feat_peer",
                      help="Models anonymously critique each other's responses")
            st.toggle("🗣️ Debate Mode", value=False, key="debate_mode",
                      help="Show raw peer review critique text in model cards")

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Trust Score Weights ───────────────────────────────────────────
            st.markdown('<div class="sb-section-label">Trust Weights</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-glass-card">', unsafe_allow_html=True)

            sem_weight = st.slider(
                "Semantic Weight", 0.0, 1.0, 0.6, step=0.05,
                key="sem_weight",
                help="Weight given to DBSCAN semantic clustering signal",
            )
            peer_weight = round(1.0 - sem_weight, 2)
            # Visual weight bar
            sem_pct = int(sem_weight * 100)
            peer_pct = 100 - sem_pct
            st.markdown(f"""
<div class="sb-weight-row">
  <div class="sb-weight-bar">
    <div class="sb-weight-fill-sem" style="width:{sem_pct}%"></div>
  </div>
  <div class="sb-weight-labels">
    <span style="color:#818cf8">Semantic {sem_pct}%</span>
    <span style="color:#f59e0b">Peer {peer_pct}%</span>
  </div>
</div>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Model Health Status ───────────────────────────────────────────
            st.markdown('<div class="sb-section-label">Model Health</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-glass-card">', unsafe_allow_html=True)

            if model_health:
                for model_name, is_healthy in model_health.items():
                    dot_class = "health-dot-ok" if is_healthy else "health-dot-bad"
                    status_text = "Online" if is_healthy else "Offline"
                    status_color = "#4ade80" if is_healthy else "#ef4444"
                    short_name = model_name[:20] + "…" if len(model_name) > 20 else model_name
                    st.markdown(f"""
<div class="sb-health-row">
  <div class="sb-health-dot {dot_class}"></div>
  <span class="sb-health-name">{short_name}</span>
  <span class="sb-health-status" style="color:{status_color}">{status_text}</span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="sb-empty-hint">Run a query to see live model health</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Free Usage Tracker ────────────────────────────────────────────
            st.markdown('<div class="sb-section-label">Free Usage Tracker</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-glass-card">', unsafe_allow_html=True)

            config = st.session_state.get("config", {})
            rate_tracker = st.session_state.get("rate_tracker")
            model_configs = {cfg["name"]: cfg for cfg in config.get("models", [])}

            env_key_map = {
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "cerebras": "CEREBRAS_API_KEY",
                "mistral": "MISTRAL_API_KEY",
                "openrouter_llama3": "OPENROUTER_API_KEY",
                "openrouter_gemma": "OPENROUTER_API_KEY",
                "nvidia_nim": "NVIDIA_NIM_API_KEY",
                "cohere": "COHERE_API_KEY",
            }

            if rate_tracker and model_configs:
                for model_name, cfg in model_configs.items():
                    daily_limit = cfg.get("daily_limit", 100)
                    used = rate_tracker.get_count(model_name)
                    remaining = max(0, daily_limit - used)
                    pct = min(100, int((used / daily_limit) * 100)) if daily_limit else 0
                    api_key_name = env_key_map.get(model_name)
                    has_key = bool(api_key_name and os.getenv(api_key_name))
                    key_icon = "✅" if has_key else "⚠️"
                    bar_color = "#ef4444" if pct > 80 else "#f59e0b" if pct > 50 else "#4ade80"
                    display_name = model_name.replace("openrouter_", "OR-").replace("_", " ").title()
                    st.markdown(f"""
<div class="sb-usage-row">
  <div class="sb-usage-header">
    <span class="sb-usage-name">{key_icon} {display_name}</span>
    <span class="sb-usage-count">{used}/{daily_limit}</span>
  </div>
  <div class="sb-usage-track">
    <div class="sb-usage-fill" style="width:{pct}%; background:{bar_color}"></div>
  </div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="sb-empty-hint">Run a query to populate usage stats</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Query History ─────────────────────────────────────────────────
            st.markdown('<div class="sb-section-label">Query History</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-glass-card">', unsafe_allow_html=True)

            history: List[str] = st.session_state.get("query_history", [])
            if not history:
                st.markdown('<div class="sb-empty-hint">No queries yet — ask something!</div>', unsafe_allow_html=True)
            else:
                for i, hist_query in enumerate(reversed(history[-15:])):
                    truncated = hist_query[:42] + "…" if len(hist_query) > 42 else hist_query
                    if st.button(
                        f"↩ {truncated}",
                        key=f"hist_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["reload_query"] = hist_query
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Session Footer ────────────────────────────────────────────────
            sess_id = st.session_state.get("session_id", "")
            st.markdown(f"""
<div class="sb-footer">
  <span>Session</span>
  <code>{sess_id[:12]}…</code>
</div>
""", unsafe_allow_html=True)

    @staticmethod
    def add_to_history(query: str) -> None:
        """Add a query to the session history list (max 20 entries)."""
        history: List[str] = st.session_state.get("query_history", [])
        if query not in history:
            history.append(query)
        if len(history) > 20:
            history = history[-20:]
        st.session_state["query_history"] = history