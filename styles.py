"""
VeritasAI — Premium Dark UI Stylesheet.
Injects custom Tailwind-like CSS overrides into Streamlit using st.markdown with unsafe_allow_html=True.
Provides responsive, glassmorphic layout renderers for all stages.
"""

import streamlit as st

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════
   1. STREAMLIT CHROME RESETS
══════════════════════════════════════════ */

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0B0E14 !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Hide default Streamlit header & footer */
#MainMenu, footer, header { visibility: hidden !important; }

/* Remove top padding Streamlit adds */
[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1200px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0B0E14; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3561; }


/* ══════════════════════════════════════════
   2. SIDEBAR — GLASSMORPHISM
══════════════════════════════════════════ */

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0a0d18 0%, #0e1220 50%, #0a0f1c 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0.75rem 0.75rem 1.5rem !important;
}

/* ── Branding ── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 4px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 14px;
}
.sb-logo-ring {
    font-size: 22px;
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(92,107,192,0.15);
    border: 1px solid rgba(92,107,192,0.3);
    border-radius: 12px;
    flex-shrink: 0;
}
.sb-brand-name {
    font-size: 15px; font-weight: 700;
    color: #e2e8f0;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.01em;
}
.sb-brand-sub {
    font-size: 10px; color: #475569;
    font-family: 'Outfit', sans-serif;
    margin-top: 1px;
}
.sb-live-badge {
    margin-left: auto;
    display: flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 3px 9px;
    font-size: 10px; font-weight: 600;
    color: #4ade80;
    font-family: 'Outfit', sans-serif;
    white-space: nowrap;
}
.sb-live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: sb-pulse 2s ease-in-out infinite;
    display: inline-block;
}
@keyframes sb-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* ── Section labels ── */
.sb-section-label {
    font-size: 9.5px; font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Outfit', sans-serif;
    margin: 14px 0 6px 2px;
}

/* ── Glass cards ── */
.sb-glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px;
    backdrop-filter: blur(10px);
    margin-bottom: 4px;
}
.sb-controls-card label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Activity monitor ── */
.sb-activity-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.sb-activity-cell {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
}
.sb-activity-value {
    font-size: 22px; font-weight: 700;
    color: #c7d2fe;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.sb-activity-desc {
    font-size: 9px; color: #475569;
    font-family: 'Outfit', sans-serif;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Weight bar ── */
.sb-weight-row { margin-top: 8px; }
.sb-weight-bar {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.sb-weight-fill-sem {
    height: 100%;
    background: linear-gradient(90deg, #818cf8, #5c6bc0);
    border-radius: 3px;
    transition: width 0.4s ease;
}
.sb-weight-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Model health rows ── */
.sb-health-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sb-health-row:last-child { border-bottom: none; }
.sb-health-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.health-dot-ok  { background: #4ade80; box-shadow: 0 0 6px rgba(74,222,128,0.5); }
.health-dot-bad { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.4); }
.sb-health-name {
    font-size: 11px; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
}
.sb-health-status {
    font-size: 10px; font-weight: 600;
    font-family: 'Outfit', sans-serif;
    flex-shrink: 0;
}

/* ── Usage tracker ── */
.sb-usage-row {
    margin-bottom: 10px;
}
.sb-usage-row:last-child { margin-bottom: 0; }
.sb-usage-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.sb-usage-name {
    font-size: 10.5px; color: #94a3b8;
    font-family: 'Outfit', sans-serif;
}
.sb-usage-count {
    font-size: 10px; color: #475569;
    font-family: 'JetBrains Mono', monospace;
}
.sb-usage-track {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.sb-usage-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* ── History buttons ── */
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #64748b !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-align: left !important;
    padding: 7px 10px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(92,107,192,0.12) !important;
    border-color: rgba(92,107,192,0.3) !important;
    color: #a5b4fc !important;
}

/* ── Empty hint ── */
.sb-empty-hint {
    font-size: 11px; color: #334155;
    font-family: 'Outfit', sans-serif;
    text-align: center;
    padding: 8px 0;
    font-style: italic;
}

/* ── Session footer ── */
.sb-footer {
    margin-top: 18px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    color: #334155;
    font-family: 'Outfit', sans-serif;
}
.sb-footer code {
    font-size: 9px;
    color: #3d4fa8;
    background: rgba(61,79,168,0.12);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Slider track inside sidebar */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
    background: #5c6bc0 !important;
}


/* ══════════════════════════════════════════
   3. MAIN SEARCH BAR
══════════════════════════════════════════ */

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #0f1219 !important;
    border: 0.5px solid #1e2535 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease !important;
    caret-color: #5c6bc0 !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #3d4fa8 !important;
    box-shadow: 0 0 0 1px #3d4fa830 !important;
    outline: none !important;
}
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {
    color: #4a5568 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Primary "Ask Everyone" run button */
[data-testid="stButton"] > button[kind="primary"] {
    background: #3d4fa8 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: background 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #4c5fbe !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    background: #2e3d8a !important;
    transform: scale(0.98) !important;
}

/* Secondary / utility buttons */
[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    border: 0.5px solid #1e2535 !important;
    border-radius: 7px !important;
    color: #64748b !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 12px !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #2d3561 !important;
    color: #a5b4fc !important;
    background: #13182a !important;
}


/* ══════════════════════════════════════════
   4. DASHBOARD PANELS
══════════════════════════════════════════ */
.vai-hero-card {
    background: rgba(12, 18, 34, 0.82);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 1.6rem 1.6rem 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.25);
}
.vai-hero-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    flex-wrap: wrap;
}
.vai-dashboard-badge {
    display: inline-flex;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: rgba(68, 82, 149, 0.16);
    color: #c3dafe;
    font-size: 0.78rem;
    font-weight: 600;
}
.vai-dashboard-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1.2rem;
}
.vai-dashboard-card {
    background: rgba(15, 20, 33, 0.88);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1rem 1rem 0.95rem;
}
.vai-dashboard-label {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.55rem;
    display: block;
}
.vai-dashboard-value {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.vai-dashboard-note {
    color: #cbd5e1;
    font-size: 0.91rem;
    line-height: 1.5;
}
@media (max-width: 960px) {
    .vai-dashboard-grid { grid-template-columns: 1fr; }
}


/* ══════════════════════════════════════════
   5. GIT DIFF — QUERY ENHANCEMENT BLOCK
══════════════════════════════════════════ */

.vai-diff-card {
    background: #0f1219;
    border: 0.5px solid #1e2535;
    border-radius: 8px;
    overflow: hidden;
    margin: 0.5rem 0 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.vai-diff-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    background: #0d1020;
    border-bottom: 0.5px solid #1e2535;
    font-size: 11px;
    color: #4a5568;
}
.vai-diff-badge {
    font-size: 10px;
    padding: 1px 8px;
    border-radius: 3px;
    font-weight: 500;
}
.vai-diff-badge.minus { background: #2d1515; color: #f87171; border: 0.5px solid #5a2020; }
.vai-diff-badge.plus  { background: #0d2318; color: #4ade80; border: 0.5px solid #1a4a30; }
.vai-diff-body { display: flex; }
.vai-diff-col  { flex: 1; padding: 10px 14px; }
.vai-diff-col + .vai-diff-col { border-left: 0.5px solid #1e2535; }
.vai-diff-line { font-size: 11px; padding: 2px 0; line-height: 1.65; white-space: pre-wrap; }
.vai-diff-line.minus   { color: #f87171; background: #1a080814; padding-left: 4px; }
.vai-diff-line.plus    { color: #4ade80; background: #081a0e14; padding-left: 4px; }
.vai-diff-line.meta    { color: #4a5568; }
.vai-diff-query-type {
    display: inline-block;
    font-size: 10px; padding: 1px 7px;
    border-radius: 3px;
    background: #131a30; color: #818cf8;
    border: 0.5px solid #2d3561;
    margin-top: 4px;
}


/* ══════════════════════════════════════════
   5. MODEL STATUS PILLS — DISPATCH TRACKER
══════════════════════════════════════════ */

.vai-dispatch-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 0.5rem 0 1rem;
}
.vai-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px;
    border-radius: 20px;
    font-family: 'Outfit', sans-serif;
    font-size: 11px;
    font-weight: 500;
    border: 0.5px solid;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.vai-pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.vai-pill-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    opacity: 0.65;
}
.vai-pill-detail {
    display: block;
    font-size: 10px;
    color: #94a3b8;
    margin-top: 2px;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* States */
.vai-pill.success {
    background: #071610; color: #4ade80; border-color: #1a4a30;
}
.vai-pill.success .vai-pill-dot { background: #4ade80; }

.vai-pill.loading {
    background: #1a1205; color: #f59e0b; border-color: #4a3008;
}
.vai-pill.loading .vai-pill-dot {
    background: #f59e0b;
    animation: pulse 1.2s ease-in-out infinite;
}

.vai-pill.failed {
    background: #1a0808; color: #f87171; border-color: #5a2020;
}
.vai-pill.failed .vai-pill-dot { background: #f87171; }

.vai-pill.skipped {
    background: #111318; color: #4a5568; border-color: #1e2535;
}
.vai-pill.skipped .vai-pill-dot { background: #374151; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}


/* ══════════════════════════════════════════
   6. CONSENSUS CARD — GLASSMORPHIC
══════════════════════════════════════════ */

.vai-consensus-card {
    background: #0d1120;
    border: 0.5px solid #2d3561;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    margin: 0.5rem 0 1rem;
}
/* top accent line */
.vai-consensus-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #5c6bc0 50%, transparent 100%);
}
.vai-consensus-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.vai-conf-badge {
    font-size: 11px; font-weight: 600;
    background: #131a30; color: #818cf8;
    border: 0.5px solid #2d3561;
    border-radius: 20px; padding: 3px 12px;
    font-family: 'Outfit', sans-serif;
}
.vai-consensus-meta {
    font-size: 11px; color: #94a3b8;
    font-family: 'Outfit', sans-serif;
}
.vai-consensus-actions {
    display: flex; gap: 6px; margin-left: auto;
}
.vai-consensus-text {
    font-size: 14px; color: #cbd5e1; line-height: 1.75;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 14px;
}
.vai-low-consensus-banner {
    background: #1a1205;
    border: 0.5px solid #4a3008;
    border-radius: 7px;
    padding: 9px 14px;
    font-size: 12px; color: #f59e0b;
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px;
}
.vai-followup-label {
    font-size: 10px; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 7px; font-family: 'Outfit', sans-serif;
}
.vai-followup-row { display: flex; flex-wrap: wrap; gap: 6px; }
.vai-followup-chip {
    font-size: 11px; color: #64748b;
    background: #0f1219; border: 0.5px solid #1e2535;
    border-radius: 20px; padding: 5px 12px;
    font-family: 'Outfit', sans-serif;
    display: inline-block;
}


/* ══════════════════════════════════════════
   7. INDIVIDUAL MODEL CARDS — GLASSMORPHIC
══════════════════════════════════════════ */

.vai-model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 15px;
    margin: 0.5rem 0 1rem;
}
.vai-model-card {
    background: #0f1219;
    border: 0.5px solid #1e2535;
    border-radius: 9px;
    padding: 15px 18px;
    transition: border-color 0.2s ease;
    font-family: 'Outfit', sans-serif;
    position: relative;
}
.vai-model-card:hover { border-color: #2d3561; }
.vai-model-card.outlier {
    border-color: #5a2020;
    background: #110a0a;
}
.vai-mc-header {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 8px;
}
.vai-mc-status-dot {
    width: 7px; height: 7px;
    border-radius: 50%; flex-shrink: 0;
}
.vai-mc-status-dot.ok    { background: #22c55e; }
.vai-mc-status-dot.warn  { background: #f59e0b; }
.vai-mc-status-dot.bad   { background: #ef4444; }
.vai-mc-status-dot.skip  { background: #374151; }
.vai-mc-name {
    font-size: 12px; font-weight: 500; color: #94a3b8;
}
.vai-mc-latency {
    font-size: 10px; color: #4a5568;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 4px;
}
.vai-mc-trust-score {
    margin-left: auto;
    font-size: 11px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.score-high { color: #4ade80; }
.score-mid  { color: #f59e0b; }
.score-low  { color: #f87171; }

.vai-trust-bar {
    height: 3px; background: #1e2535;
    border-radius: 2px; margin-bottom: 10px; overflow: hidden;
}
.vai-trust-bar-fill {
    height: 3px; border-radius: 2px;
    transition: width 0.5s ease;
}
.fill-high { background: #22c55e; }
.fill-mid  { background: #f59e0b; }
.fill-low  { background: #ef4444; }

.vai-mc-response {
    font-size: 13px; color: #cbd5e1; line-height: 1.6;
    white-space: pre-wrap;
}

.vai-outlier-warning {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10px; color: #f87171;
    background: #1a0808; border: 0.5px solid #5a2020;
    border-radius: 4px; padding: 3px 8px; margin-top: 8px;
}
.vai-peer-rank {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10px; color: #818cf8;
    background: #131a30; border: 0.5px solid #2d3561;
    border-radius: 4px; padding: 3px 8px; margin-top: 5px; margin-left: 5px;
}


/* ══════════════════════════════════════════
   8. METRICS / STAT CARDS
══════════════════════════════════════════ */

[data-testid="stMetric"] {
    background: #0f1219 !important;
    border: 0.5px solid #1e2535 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetric"] label {
    color: #4a5568 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 22px !important;
    font-weight: 500 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}


/* ══════════════════════════════════════════
   9. DATAFRAMES / TABLES
══════════════════════════════════════════ */

[data-testid="stDataFrame"] {
    background: #0f1219 !important;
    border: 0.5px solid #1e2535 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: #0d1020 !important;
    color: #4a5568 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    border-bottom: 0.5px solid #1e2535 !important;
    padding: 8px 12px !important;
}
[data-testid="stDataFrame"] td {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-bottom: 0.5px solid #1a1f2e !important;
    padding: 7px 12px !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #13182a !important;
}


/* ══════════════════════════════════════════
   10. SELECT BOX / RADIO / MISC INPUTS
══════════════════════════════════════════ */

[data-testid="stSelectbox"] > div > div {
    background: #0f1219 !important;
    border: 0.5px solid #1e2535 !important;
    border-radius: 7px !important;
    color: #e2e8f0 !important;
}
[data-testid="stRadio"] label { color: #94a3b8 !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f1219 !important;
    border: 0.5px dashed #2d3561 !important;
    border-radius: 8px !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div {
    background: #1e2535 !important;
    border-radius: 3px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: #5c6bc0 !important;
    border-radius: 3px !important;
}

/* Info / Warning / Error alerts */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 7px !important;
    font-size: 13px !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 0.5px solid #1e2535 !important;
    margin: 1rem 0 !important;
}

/* Section headers in main content */
h1 { font-size: 24px !important; font-weight: 600 !important; color: #e2e8f0 !important; margin-bottom: 1.25rem !important; font-family: 'Outfit', sans-serif !important; }
h2 { font-size: 18px !important; font-weight: 500 !important; color: #94a3b8 !important; margin-bottom: 1rem !important; font-family: 'Outfit', sans-serif !important; }
h3 {
    font-size: 11px !important; font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    margin: 1.2rem 0 0.4rem !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Spinner */
[data-testid="stSpinner"] > div > div {
    border-top-color: #5c6bc0 !important;
}

/* Caption / small text */
[data-testid="stCaptionContainer"] {
    color: #4a5568 !important;
    font-size: 11px !important;
}
"""


def render_diff(original: str, enhanced: str, query_type: str = "factual") -> str:
    """Render a Git-diff style query enhancement block."""
    orig_lines = "\n".join(
        f'<div class="vai-diff-line minus">− {line}</div>'
        for line in original.strip().splitlines() or [original.strip()]
    )
    enh_lines = "\n".join(
        f'<div class="vai-diff-line plus">+ {line}</div>'
        for line in enhanced.strip().splitlines() or [enhanced.strip()]
    )
    return f"""
<div class="vai-diff-card">
  <div class="vai-diff-header">
    <span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#cbd5e1">query_optimization.diff</span>
    <span class="vai-diff-badge minus">− original</span>
    <span class="vai-diff-badge plus">+ enhanced</span>
  </div>
  <div class="vai-diff-body">
    <div class="vai-diff-col">{orig_lines}</div>
    <div class="vai-diff-col">
      {enh_lines}
      <div class="vai-diff-line meta" style="margin-top:4px">
        <span class="vai-diff-query-type">type: {query_type}</span>
      </div>
    </div>
  </div>
</div>
"""


def render_dispatch_pills(model_statuses: dict) -> str:
    """
    Render the operational dispatch tracker row.
    model_statuses: dict of { model_name: {"status": "success"|"loading"|"failed"|"skipped", "latency_ms": int|None, "detail": str|None} }
    """
    pills_html = ""
    for name, info in model_statuses.items():
        status = info.get("status", "skipped")
        lat = info.get("latency_ms")
        detail = info.get("detail")
        lat_str = f"{lat}ms" if lat else ("…" if status == "loading" else "—")
        detail_html = f"<span class='vai-pill-detail'>{detail}</span>" if detail else ""
        pills_html += f"""
<div class="vai-pill {status}">
  <div class="vai-pill-dot"></div>
  {name}
  <span class="vai-pill-time">{lat_str}</span>
  {detail_html}
</div>"""
    return f'<div class="vai-dispatch-row">{pills_html}</div>'


def render_consensus_card(
    answer: str,
    confidence_pct: int,
    models_used: int,
    models_total: int,
    follow_ups: list[str] | None = None,
    low_consensus: bool = False,
    high_dissent: bool = False,
) -> str:
    """Render the glassmorphic consensus answer card."""
    warning_html = ""
    if models_used < 2:
        warning_html = """
<div class="vai-low-consensus-banner" style="background:#2b1a08; color:#f59e0b; border-color:#4a3008;">
  ⚠ Only one model was available to respond. Consensus confidence is unavailable. Review the single response below.
</div>"""
    elif high_dissent:
        warning_html = """
<div class="vai-low-consensus-banner" style="background:#2d1515; color:#ef4444; border-color:#5a2020;">
  ❌ High Dissent Warning — The active models returned highly contradictory viewpoints. Real-time consensus could not be safely reached. Inspect individual model outputs below.
</div>"""
    elif low_consensus:
        warning_html = """
<div class="vai-low-consensus-banner">
  ⚠ Warning: Low consensus detected — models have divergent viewpoints. Review individual responses below.
</div>"""

    # Confidence meter bar (visual confidence meter)
    meter_html = ""
    if models_used >= 2 and not high_dissent:
        if confidence_pct >= 70:
            bar_color = "#10B981"  # Emerald
        elif confidence_pct >= 50:
            bar_color = "#F59E0B"  # Amber
        else:
            bar_color = "#EF4444"  # Rose
            
        meter_html = f"""
<div style="margin: 0.5rem 0 1rem 0;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
    <span style="font-size: 11px; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Consensus Strength</span>
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: {bar_color};">{confidence_pct}%</span>
  </div>
  <div style="width: 100%; height: 6px; background: #1e2535; border-radius: 3px; overflow: hidden; border: 0.5px solid #1e2535;">
    <div style="width: {confidence_pct}%; height: 100%; background: {bar_color}; border-radius: 3px;"></div>
  </div>
</div>
"""

    fu_html = ""
    if follow_ups and not high_dissent:
        chips = "".join(f'<span class="vai-followup-chip">{q}</span>' for q in follow_ups[:3])
        fu_html = f"""
<div class="vai-followup-label">Suggested follow-ups</div>
<div class="vai-followup-row">{chips}</div>"""

    return f"""
<div class="vai-consensus-card">
  <div class="vai-consensus-header">
    <span class="vai-conf-badge">{confidence_pct}% consensus rating</span>
    <span class="vai-consensus-meta">Calculated across {models_used} / {models_total} models</span>
  </div>
  {warning_html}
  {meter_html}
  <div class="vai-consensus-text">{answer}</div>
  {fu_html}
</div>
"""


def render_model_card_html(
    name: str,
    response: str,
    trust_score: float,
    latency_ms: int | None = None,
    peer_rank_score: float | None = None,
    is_outlier: bool = False,
    status: str = "success",
    semantic_score: float | None = None,
) -> str:
    """Render a single model response card as HTML."""
    if status == "skipped":
        return f"""
<div class="vai-model-card" style="opacity: 0.5;">
  <div class="vai-mc-header">
    <div class="vai-mc-status-dot skip"></div>
    <span class="vai-mc-name">{name}</span>
    <span class="vai-mc-latency">—</span>
    <span class="vai-mc-trust-score score-low">Skipped</span>
  </div>
  <div class="vai-trust-bar">
    <div class="vai-trust-bar-fill fill-low" style="width:0%"></div>
  </div>
  <div class="vai-mc-response" style="color: #4a5568; font-style: italic;">API key not configured or model deactivated.</div>
</div>
        """
    elif status == "failed":
        reason = response or "Unknown connection error"
        return f"""
<div class="vai-model-card" style="border-color: #5a2020; background: #1a0808;">
  <div class="vai-mc-header">
    <div class="vai-mc-status-dot bad"></div>
    <span class="vai-mc-name">{name}</span>
    <span class="vai-mc-latency">{f"{latency_ms}ms" if latency_ms else "Error"}</span>
    <span class="vai-mc-trust-score score-low">Failed</span>
  </div>
  <div class="vai-trust-bar">
    <div class="vai-trust-bar-fill fill-low" style="width:0%"></div>
  </div>
  <div class="vai-mc-response" style="color: #f87171; font-family: monospace;">{reason}</div>
</div>
        """

    # Derive colour classes from trust score
    if trust_score >= 0.70:
        score_class, fill_class = "score-high", "fill-high"
        dot_class = "ok"
    elif trust_score >= 0.50:
        score_class, fill_class = "score-mid", "fill-mid"
        dot_class = "warn"
    else:
        score_class, fill_class = "score-low", "fill-low"
        dot_class = "bad"

    bar_width = int(trust_score * 100)
    lat_str = f"{latency_ms}ms" if latency_ms else ""
    outlier_html = ""
    if is_outlier:
        outlier_html = '<div class="vai-outlier-warning">⚠ Outlier: High variance from consensus</div>'

    # Build breakdown HTML (Visual Trust Breakdown)
    breakdown_html = ""
    if semantic_score is not None and peer_rank_score is not None:
        breakdown_html = f"""
  <div style="font-size: 10px; color: #64748b; margin-top: 4px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; display: flex; gap: 8px;">
    <span>Breakdown:</span>
    <span style="color: #a7f3d0;">Semantic Sim: {semantic_score:.2f} (60%)</span>
    <span style="color: #64748b;">|</span>
    <span style="color: #c7d2fe;">Peer Rank: {peer_rank_score:.2f} (40%)</span>
  </div>
"""

    peer_html = ""
    if peer_rank_score is not None:
        peer_html = f'<span class="vai-peer-rank">peer rank: {peer_rank_score:.2f}</span>'

    card_class = "vai-model-card outlier" if is_outlier else "vai-model-card"
    return f"""
<div class="{card_class}">
  <div class="vai-mc-header">
    <div class="vai-mc-status-dot {dot_class}"></div>
    <span class="vai-mc-name">{name}</span>
    <span class="vai-mc-latency">{lat_str}</span>
    <span class="vai-mc-trust-score {score_class}">{trust_score:.2f}</span>
  </div>
  <div class="vai-trust-bar">
    <div class="vai-trust-bar-fill {fill_class}" style="width:{bar_width}%"></div>
  </div>
  {breakdown_html}
  <details style="margin-top: 8px; cursor: pointer;">
    <summary style="font-size: 11px; color: #64748b; outline: none; margin-bottom: 6px; font-weight: 500;">
      View Raw Response
    </summary>
    <div class="vai-mc-response" style="cursor: auto;">{response}</div>
  </details>
  {outlier_html}
  {peer_html}
</div>
"""


def render_model_grid(model_results: list[dict]) -> str:
    """
    Render the full model response grid.
    Each dict in model_results should contain render_model_card_html parameters.
    """
    cards = "".join(render_model_card_html(**m) for m in model_results)
    return f'<div class="vai-model-grid">{cards}</div>'


def inject_styles() -> None:
    """Inject global CSS into the Streamlit app view container."""
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)


ADMIN_CSS = """
/* Admin page extras */
[data-testid="stTextInput"][aria-label*="password"] input {
    letter-spacing: 0.25em !important;
}

/* Plotly chart overrides to match the deep dark grid */
.js-plotly-plot .plotly .bg {
    fill: #0f1219 !important;
}
.js-plotly-plot .gridlayer path {
    stroke: #1e2535 !important;
}
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text {
    fill: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}
.js-plotly-plot .gtitle {
    fill: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}
"""

def inject_admin_styles() -> None:
    """Inject extra admin panel CSS styling overrides."""
    st.markdown(f"<style>{ADMIN_CSS}</style>", unsafe_allow_html=True)


def render_reliability_card(
    models_consulted: int,
    models_agreeing: int,
    outliers: int,
    reliability_level: str,
) -> str:
    """Render the system reliability card."""
    level_colors = {
        "High": {"text": "#10B981", "bg": "rgba(16, 185, 129, 0.03)", "border": "rgba(16, 185, 129, 0.2)"},
        "Moderate": {"text": "#F59E0B", "bg": "rgba(245, 158, 11, 0.03)", "border": "rgba(245, 158, 11, 0.2)"},
        "Low": {"text": "#EF4444", "bg": "rgba(239, 68, 68, 0.03)", "border": "rgba(239, 68, 68, 0.2)"},
        "Unavailable": {"text": "#64748B", "bg": "rgba(148, 163, 184, 0.03)", "border": "rgba(148, 163, 184, 0.2)"}
    }
    cfg = level_colors.get(reliability_level, level_colors["Unavailable"])
    
    return f"""
<div class="vai-consensus-card" style="border-color: {cfg['border']}; background: {cfg['bg']}; margin-top: 0.5rem; position: relative;">
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap;">
    <h4 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 13px; font-weight: 600; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.05em;">⚖️ Truth Engine Reliability</h4>
    <span style="font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 20px; background: {cfg['border']}; color: {cfg['text']};">{reliability_level} Reliability</span>
  </div>
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; text-align: center;">
    <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; border: 0.5px solid #1e2535;">
      <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">Consulted</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 600; color: #e2e8f0;">{models_consulted}</div>
    </div>
    <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; border: 0.5px solid #1e2535;">
      <div style="background: transparent; font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">In Agreement</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 600; color: #e2e8f0;">{models_agreeing}</div>
    </div>
    <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; border: 0.5px solid #1e2535;">
      <div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">Outliers</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 600; color: { '#ef4444' if outliers > 0 else '#64748b' };">{outliers}</div>
    </div>
  </div>
</div>
"""
