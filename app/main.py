"""
DatumMind Analyst — Premium Dark-Theme Dashboard
AI-Powered Data Analysis Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    APP_TITLE, APP_ICON, APP_SUBTITLE, APP_DESCRIPTION, APP_VERSION,
    PAGE_CONFIG, SAMPLE_QUERIES, SUPPORTED_EXTENSIONS,
    LLM_PROVIDERS, DEFAULT_PROVIDER, TEMPERATURE, MAX_RETRIES,
    GROQ_API_KEY, HF_API_KEY, GOOGLE_API_KEY, THEME,
)
from app.ui_components import (
    inject_theme, render_header, render_footer,
    render_metric_card, render_section_header, render_status_badge,
    render_plan, render_code, render_results, render_charts,
    render_insights, render_status_messages, render_error,
    render_dataset_info, render_kpi_row, render_column_stats,
    render_query_card, render_sidebar_divider,
)
from analysis.data_utils import load_dataset, auto_clean, get_schema, format_schema_for_llm
from analysis.viz_utils import set_plot_style, close_all_figures


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)

# ── Inject Theme & Plot Style ─────────────────────────────────────────────────
inject_theme()
set_plot_style()


# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "agent": None,
        "datasets": {},
        "provider_name": DEFAULT_PROVIDER,
        "api_key": "",
        "model_name": "",
        "processing": False,
        "agent_config_hash": "",
        "query_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Safety: reset stale provider name from old sessions
if st.session_state.provider_name not in LLM_PROVIDERS:
    st.session_state.provider_name = DEFAULT_PROVIDER


def _env_key(provider_name: str) -> str:
    p = LLM_PROVIDERS[provider_name]
    if p["key"] == "groq": return GROQ_API_KEY
    if p["key"] == "huggingface": return HF_API_KEY
    if p["key"] == "gemini": return GOOGLE_API_KEY
    return ""


def _config_hash(pn, ak, mn):
    return f"{pn}|{ak[:8] if ak else ''}|{mn}"


def get_agent(provider_name, api_key, model_name):
    h = _config_hash(provider_name, api_key, model_name)
    if st.session_state.agent is None or st.session_state.agent_config_hash != h:
        if not api_key:
            return None
        try:
            from agent.core import DatumMindAgent
            pk = LLM_PROVIDERS[provider_name]["key"]
            st.session_state.agent = DatumMindAgent(
                provider_key=pk, api_key=api_key, model_name=model_name,
                temperature=TEMPERATURE, max_retries=MAX_RETRIES,
            )
            st.session_state.agent_config_hash = h
            # Re-register datasets: first as "df", rest as "df2", "df3", etc.
            for idx, (name, df) in enumerate(st.session_state.datasets.items()):
                var_name = "df" if idx == 0 else f"df{idx + 1}"
                st.session_state.agent.load_dataset(var_name, df)
        except Exception as e:
            st.error(f"Agent init failed: {e}")
            return None
    return st.session_state.agent


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        display: flex; align-items: center; gap: 12px;
        padding: 8px 0 4px 0;
    ">
        <div style="
            width: 36px; height: 36px;
            background: {THEME['accent_gradient']};
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px;
            box-shadow: {THEME['glow_primary']};
        ">🧠</div>
        <div>
            <p style="margin:0; font-size: 1rem; font-weight: 700; color: {THEME['text_primary']};">DatumMind</p>
            <p style="margin:0; font-size: 0.68rem; color: {THEME['text_muted']};">AI Data Analyst · v{APP_VERSION}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar_divider()

    # ── Provider & API Key (Hidden) ───────────────────────────────────
    provider_name = "Groq (Free)"
    st.session_state.provider_name = provider_name
    pcfg = LLM_PROVIDERS[provider_name]

    st.markdown(f"<p style='color:{THEME['text_muted']};font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>🤖 Model</p>", unsafe_allow_html=True)
    model_name = st.selectbox("Model", options=pcfg["models"], index=0, label_visibility="collapsed")

    # Use environment variable for API key (hidden from UI)
    api_key = _env_key(provider_name)
    st.session_state.api_key = api_key

    render_sidebar_divider()

    # ── Dataset Upload ────────────────────────────────────────────────
    st.markdown(f"<p style='color:{THEME['text_muted']};font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>📂 Datasets</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload", type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            if uf.name not in st.session_state.datasets:
                try:
                    df = auto_clean(load_dataset(uf))
                    st.session_state.datasets[uf.name] = df
                    agent = get_agent(provider_name, api_key, model_name)
                    if agent:
                        if len(st.session_state.datasets) == 1:
                            agent.load_dataset("df", df)
                        else:
                            agent.load_dataset(f"df{len(st.session_state.datasets)}", df)
                            first = list(st.session_state.datasets.values())[0]
                            agent.load_dataset("df", first)
                    st.success(f"✅ {uf.name}")
                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")

    if st.button("📊 Load Sample Dataset", use_container_width=True):
        sp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "sales_sample.csv")
        if os.path.exists(sp):
            df = auto_clean(pd.read_csv(sp))
            st.session_state.datasets["sales_sample.csv"] = df
            agent = get_agent(provider_name, api_key, model_name)
            if agent:
                agent.load_dataset("df", df)
            st.rerun()

    # Show loaded datasets
    if st.session_state.datasets:
        for name, df in st.session_state.datasets.items():
            render_dataset_info(df, name)

    render_sidebar_divider()

    # ── Controls ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.rerun()
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["messages", "datasets", "agent", "agent_config_hash", "query_history"]:
                st.session_state[k] = [] if k in ("messages", "query_history") else {} if k == "datasets" else None if k == "agent" else ""
            st.rerun()

    # ── Query History ─────────────────────────────────────────────────
    if st.session_state.query_history:
        render_sidebar_divider()
        st.markdown(f"<p style='color:{THEME['text_muted']};font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>🕘 Recent Queries</p>", unsafe_allow_html=True)
        for q in st.session_state.query_history[-5:]:
            st.markdown(f"<p style='color:{THEME['text_secondary']};font-size:0.75rem;padding:4px 0;border-bottom:1px solid {THEME['border']}22;'>• {q[:60]}{'…' if len(q)>60 else ''}</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

render_header()

has_data = bool(st.session_state.datasets)
has_key = bool(st.session_state.api_key)

# ── Status Bar ────────────────────────────────────────────────────────────────
status_cols = st.columns([2, 2, 2, 6])
with status_cols[0]:
    if has_key:
        st.markdown(render_status_badge("● API Connected", "success"), unsafe_allow_html=True)
    else:
        st.markdown(render_status_badge("○ API Key Missing (.env)", "warning"), unsafe_allow_html=True)
with status_cols[1]:
    if has_data:
        n = len(st.session_state.datasets)
        st.markdown(render_status_badge(f"● {n} Dataset{'s' if n > 1 else ''}", "success"), unsafe_allow_html=True)
    else:
        st.markdown(render_status_badge("○ No Data", "warning"), unsafe_allow_html=True)
with status_cols[2]:
    st.markdown(render_status_badge(f"🤖 {provider_name.split('(')[0].strip()}", "info"), unsafe_allow_html=True)

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

# ── No Data State ─────────────────────────────────────────────────────────────
if not has_data:
    st.markdown(f"""
    <div style="
        background: {THEME['bg_surface']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
        margin-top: 20px;
    ">
        <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.6;">📊</div>
        <h2 style="color: {THEME['text_primary']}; font-weight: 700; margin-bottom: 8px;">
            Welcome to DatumMind Analyst
        </h2>
        <p style="color: {THEME['text_muted']}; font-size: 0.95rem; max-width: 500px; margin: 0 auto 32px;">
            Upload your dataset or load the sample data to start analyzing with AI.
        </p>
        <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
            <div style="text-align: center; max-width: 180px;">
                <div style="
                    width: 52px; height: 52px; margin: 0 auto 12px;
                    background: {THEME['accent_primary']}15;
                    border: 1px solid {THEME['accent_primary']}30;
                    border-radius: 14px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.4rem;
                ">📂</div>
                <p style="color: {THEME['text_primary']}; font-weight: 600; font-size: 0.88rem; margin-bottom: 4px;">Upload Data</p>
                <p style="color: {THEME['text_muted']}; font-size: 0.75rem;">CSV or Excel files</p>
            </div>
            <div style="text-align: center; max-width: 180px;">
                <div style="
                    width: 52px; height: 52px; margin: 0 auto 12px;
                    background: {THEME['accent_secondary']}15;
                    border: 1px solid {THEME['accent_secondary']}30;
                    border-radius: 14px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.4rem;
                ">💬</div>
                <p style="color: {THEME['text_primary']}; font-weight: 600; font-size: 0.88rem; margin-bottom: 4px;">Ask Questions</p>
                <p style="color: {THEME['text_muted']}; font-size: 0.75rem;">Natural language</p>
            </div>
            <div style="text-align: center; max-width: 180px;">
                <div style="
                    width: 52px; height: 52px; margin: 0 auto 12px;
                    background: {THEME['success']}15;
                    border: 1px solid {THEME['success']}30;
                    border-radius: 14px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.4rem;
                ">🧠</div>
                <p style="color: {THEME['text_primary']}; font-weight: 600; font-size: 0.88rem; margin-bottom: 4px;">Get Insights</p>
                <p style="color: {THEME['text_muted']}; font-size: 0.75rem;">Charts & analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ══════════════════════════════════════════════════════════════════
    #  TABBED DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    tab_overview, tab_analysis, tab_explore = st.tabs(["📊 Overview", "🧠 AI Analysis", "🔍 Explore Data"])

    # Get the first loaded dataset for overview/explore
    first_df_name = list(st.session_state.datasets.keys())[0]
    first_df = st.session_state.datasets[first_df_name]

    # ── TAB: Overview ─────────────────────────────────────────────────
    with tab_overview:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        # KPI Cards
        render_kpi_row(first_df)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Dataset preview and column stats side by side
        col_left, col_right = st.columns([3, 2])

        with col_left:
            render_section_header("Data Preview", f"First 10 rows of {first_df_name}", "📋")
            st.dataframe(first_df.head(10), width="stretch", height=350)

        with col_right:
            render_section_header("Column Statistics", f"{len(first_df.columns)} columns", "📊")
            render_column_stats(first_df)

        # Quick stats row
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        numeric_df = first_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            render_section_header("Numeric Summary", "Descriptive statistics for numeric columns", "📐")
            desc = numeric_df.describe().T
            desc = desc.round(2)
            st.dataframe(desc, width="stretch", height=min(350, 40 + len(desc) * 35))

    # ── TAB: AI Analysis ──────────────────────────────────────────────
    with tab_analysis:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        # Sample query cards (only when no messages yet)
        if not st.session_state.messages:
            render_section_header("Quick Start", "Click a query or type your own below", "⚡")
            q_cols = st.columns(2)
            for i, (icon, query) in enumerate(SAMPLE_QUERIES[:6]):
                with q_cols[i % 2]:
                    if render_query_card(icon, query, f"sq_{i}"):
                        st.session_state.messages.append({"role": "user", "content": query})
                        st.session_state.query_history.append(query)
                        st.rerun()

        # Chat History
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["content"]:
                    st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    if msg.get("plan"):
                        render_plan(msg["plan"], msg.get("plan_text", ""))
                    if msg.get("code"):
                        render_code(msg["code"])
                    if msg.get("execution"):
                        render_results(msg["execution"])
                        render_charts(msg["execution"])
                    if msg.get("narrative"):
                        render_insights(msg["narrative"])
                    if msg.get("status_messages"):
                        with st.expander("🔍 Pipeline Steps", expanded=False):
                            render_status_messages(msg["status_messages"])
                    if msg.get("error") and not msg.get("narrative"):
                        render_error(msg["error"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about your data…", disabled=st.session_state.processing):
            if not has_key:
                st.error("⚠️ Server API key missing. Please check the backend configuration (.env file).")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.query_history.append(prompt)

            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                agent = get_agent(provider_name, api_key, model_name)
                if not agent:
                    err = f"Agent init failed. {pcfg['help']}"
                    render_error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "error": err})
                    st.stop()

                status_ctr = st.empty()
                status_msgs = []

                def _cb(m):
                    status_msgs.append(m)
                    status_ctr.markdown(
                        f"<div style='color:{THEME['text_muted']};font-size:0.8rem;'>"
                        + " → ".join(status_msgs[-3:]) + "</div>",
                        unsafe_allow_html=True,
                    )

                st.session_state.processing = True
                try:
                    close_all_figures()
                    resp = agent.analyze(prompt, status_callback=_cb)
                finally:
                    st.session_state.processing = False

                status_ctr.empty()

                amsg = {
                    "role": "assistant", "content": "",
                    "plan": resp.plan, "plan_text": resp.plan_text,
                    "code": resp.code, "execution": resp.execution,
                    "narrative": resp.narrative, "status_messages": resp.status_messages,
                    "error": resp.error,
                }

                if resp.error and not resp.narrative:
                    render_error(resp.error)
                    amsg["content"] = f"Error: {resp.error}"
                else:
                    if resp.plan:
                        render_plan(resp.plan, resp.plan_text)
                    if resp.code:
                        render_code(resp.code)
                    if resp.execution:
                        render_results(resp.execution)
                        render_charts(resp.execution)
                    if resp.narrative:
                        render_insights(resp.narrative)
                        amsg["content"] = resp.narrative[:200] + "…"
                    with st.expander("🔍 Pipeline Steps", expanded=False):
                        render_status_messages(resp.status_messages)

                st.session_state.messages.append(amsg)

    # ── TAB: Explore Data ─────────────────────────────────────────────
    with tab_explore:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        # Dataset selector if multiple
        if len(st.session_state.datasets) > 1:
            explore_name = st.selectbox(
                "Select dataset", list(st.session_state.datasets.keys()),
                label_visibility="collapsed",
            )
            explore_df = st.session_state.datasets[explore_name]
        else:
            explore_name = first_df_name
            explore_df = first_df

        render_section_header("Interactive Explorer", f"Browse & filter {explore_name}", "🔍")

        # Column filter
        all_cols = explore_df.columns.tolist()
        selected_cols = st.multiselect(
            "Select columns to display",
            all_cols, default=all_cols[:8] if len(all_cols) > 8 else all_cols,
            label_visibility="collapsed",
        )

        if selected_cols:
            filtered_df = explore_df[selected_cols]

            # Search/filter
            search = st.text_input("🔎 Search/filter rows…", placeholder="Type to filter across all columns", label_visibility="collapsed")
            if search:
                mask = filtered_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                filtered_df = filtered_df[mask]
                st.markdown(f"{render_status_badge(f'{len(filtered_df)} rows match', 'info')}", unsafe_allow_html=True)

            st.dataframe(filtered_df, width="stretch", height=500)

            # Quick charts for numeric columns
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                render_section_header("Quick Charts", "Select a column to visualize", "📈")
                chart_col = st.selectbox("Column", numeric_cols, label_visibility="collapsed")
                chart_type = st.radio("Chart type", ["Histogram", "Box Plot"], horizontal=True, label_visibility="collapsed")

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor(THEME['bg_surface'])
                ax.set_facecolor(THEME['bg_elevated'])

                if chart_type == "Histogram":
                    ax.hist(filtered_df[chart_col].dropna(), bins=30, color=THEME['accent_primary'], edgecolor=THEME['border'], alpha=0.85)
                else:
                    bp = ax.boxplot(filtered_df[chart_col].dropna(), patch_artist=True, vert=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor(THEME['accent_primary'])
                        patch.set_alpha(0.7)

                ax.set_title(f"{chart_type} — {chart_col}", color=THEME['text_primary'], fontweight='bold')
                ax.tick_params(colors=THEME['text_muted'])
                for spine in ax.spines.values():
                    spine.set_color(THEME['border'])
                ax.grid(color=THEME['border'], alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ── Footer ────────────────────────────────────────────────────────────────────
render_footer()
