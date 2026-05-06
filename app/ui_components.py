"""
Premium dark-theme UI components for DatumMind Analyst.
Card layouts, metric displays, status badges, chart containers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.config import THEME
from analysis.viz_utils import save_figure_to_bytes


# ── CSS Theme ─────────────────────────────────────────────────────────────────

PREMIUM_CSS = f"""
<style>
    /* ── Import Font ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────────────── */
    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    .stApp {{
        background: {THEME['bg_primary']} !important;
    }}

    /* ── Hide Defaults ───────────────────────────────────────────────── */
    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {THEME['bg_surface']} !important;
        border-right: 1px solid {THEME['border']} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {{
        color: {THEME['text_secondary']} !important;
        font-size: 0.85rem;
    }}

    /* ── Tabs ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {THEME['bg_surface']};
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {THEME['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {THEME['text_muted']};
        font-weight: 500;
        font-size: 0.9rem;
        padding: 8px 20px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {THEME['accent_primary']} !important;
        color: white !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ── Expanders ───────────────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {THEME['bg_surface']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
        color: {THEME['text_primary']} !important;
        font-weight: 600 !important;
    }}
    .streamlit-expanderContent {{
        background: {THEME['bg_surface']} !important;
        border: 1px solid {THEME['border']} !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }}

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button {{
        background: {THEME['bg_elevated']} !important;
        color: {THEME['text_primary']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        padding: 6px 16px !important;
    }}
    .stButton > button:hover {{
        border-color: {THEME['accent_primary']} !important;
        box-shadow: {THEME['glow_primary']} !important;
        background: {THEME['bg_surface_hover']} !important;
    }}

    /* ── Primary Action Button ────────────────────────────────────────── */
    .stButton > button[kind="primary"] {{
        background: {THEME['accent_primary']} !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.3) !important;
    }}

    /* ── Chat Input ──────────────────────────────────────────────────── */
    .stChatInput {{
        border-color: {THEME['border']} !important;
    }}
    .stChatInput > div {{
        background: {THEME['bg_surface']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 12px !important;
    }}
    .stChatInput textarea {{
        color: {THEME['text_primary']} !important;
    }}

    /* ── Chat Messages ───────────────────────────────────────────────── */
    .stChatMessage {{
        background: {THEME['bg_surface']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }}

    /* ── Text Input ──────────────────────────────────────────────────── */
    .stTextInput > div > div {{
        background: {THEME['bg_elevated']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 8px !important;
        color: {THEME['text_primary']} !important;
    }}
    .stTextInput > div > div:focus-within {{
        border-color: {THEME['accent_primary']} !important;
        box-shadow: {THEME['glow_primary']} !important;
    }}

    /* ── Select Box ──────────────────────────────────────────────────── */
    .stSelectbox > div > div {{
        background: {THEME['bg_elevated']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 8px !important;
        color: {THEME['text_primary']} !important;
    }}

    /* ── File Uploader ───────────────────────────────────────────────── */
    .stFileUploader > div {{
        background: {THEME['bg_elevated']} !important;
        border: 1px dashed {THEME['border_accent']} !important;
        border-radius: 10px !important;
    }}

    /* ── DataFrame ───────────────────────────────────────────────────── */
    .stDataFrame {{
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
        overflow: hidden;
    }}

    /* ── Code Block ──────────────────────────────────────────────────── */
    .stCodeBlock {{
        border: 1px solid {THEME['border']} !important;
        border-radius: 10px !important;
    }}

    /* ── Download Button ─────────────────────────────────────────────── */
    .stDownloadButton > button {{
        background: transparent !important;
        border: 1px solid {THEME['border']} !important;
        color: {THEME['accent_secondary']} !important;
        font-size: 0.8rem !important;
    }}
    .stDownloadButton > button:hover {{
        border-color: {THEME['accent_secondary']} !important;
        box-shadow: {THEME['glow_accent']} !important;
    }}

    /* ── Alerts ───────────────────────────────────────────────────────── */
    .stAlert {{
        border-radius: 10px !important;
        border: 1px solid {THEME['border']} !important;
    }}

    /* ── Scrollbar ────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {THEME['bg_primary']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {THEME['border_accent']};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {THEME['text_muted']};
    }}

    /* ── Divider ──────────────────────────────────────────────────────── */
    hr {{
        border-color: {THEME['border']} !important;
        opacity: 0.5;
    }}
</style>
"""


# ── Component Functions ───────────────────────────────────────────────────────

def inject_theme():
    """Inject the premium dark theme CSS."""
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


def render_header():
    """Render the top navbar / header."""
    st.markdown(f"""
    <div style="
        background: {THEME['bg_surface']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 20px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="
                width: 44px; height: 44px;
                background: {THEME['accent_gradient']};
                border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 22px;
                box-shadow: {THEME['glow_primary']};
            ">🧠</div>
            <div>
                <h1 style="
                    margin: 0; padding: 0;
                    font-size: 1.6rem; font-weight: 800;
                    background: {THEME['accent_gradient']};
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    letter-spacing: -0.5px;
                ">DatumMind Analyst</h1>
                <p style="
                    margin: 0; padding: 0;
                    font-size: 0.8rem;
                    color: {THEME['text_muted']};
                    font-weight: 400;
                ">AI-Powered Data Analysis Agent</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="
                background: {THEME['bg_elevated']};
                border: 1px solid {THEME['border']};
                border-radius: 20px;
                padding: 4px 14px;
                font-size: 0.75rem;
                color: {THEME['text_muted']};
                font-weight: 500;
            ">v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "", icon: str = "📊", color: str = ""):
    """Render a single KPI metric card."""
    accent = color or THEME['accent_primary']
    delta_html = ""
    if delta:
        delta_color = THEME['success'] if not delta.startswith("-") else THEME['error']
        delta_html = f'<br><span style="color: {delta_color}; font-size: 0.75rem; font-weight: 500;">{delta}</span>'

    html = (
        f'<div style="background:{THEME["bg_surface"]};border:1px solid {THEME["border"]};'
        f'border-radius:14px;padding:20px;border-left:3px solid {accent};">'
        f'<span style="float:right;font-size:1.5rem;opacity:0.7;">{icon}</span>'
        f'<p style="margin:0 0 6px 0;color:{THEME["text_muted"]};font-size:0.78rem;'
        f'font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">{label}</p>'
        f'<p style="margin:0;color:{THEME["text_primary"]};font-size:1.6rem;'
        f'font-weight:700;letter-spacing:-0.5px;">{value}</p>'
        f'{delta_html}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a section header with optional subtitle."""
    st.markdown(f"""
    <div style="margin: 24px 0 16px 0;">
        <h3 style="
            margin: 0; padding: 0;
            color: {THEME['text_primary']};
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        ">{icon} {title}</h3>
        {"<p style='margin: 4px 0 0 0; color: " + THEME['text_muted'] + "; font-size: 0.8rem;'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(text: str, status: str = "info"):
    """Render a colored status badge."""
    colors = {
        "success": (THEME['success'], "rgba(16, 185, 129, 0.1)"),
        "warning": (THEME['warning'], "rgba(245, 158, 11, 0.1)"),
        "error": (THEME['error'], "rgba(239, 68, 68, 0.1)"),
        "info": (THEME['accent_secondary'], "rgba(34, 211, 238, 0.1)"),
        "active": (THEME['accent_primary'], "rgba(99, 102, 241, 0.1)"),
    }
    fg, bg = colors.get(status, colors["info"])
    return f"""<span style="
        background: {bg};
        color: {fg};
        border: 1px solid {fg}33;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    ">{text}</span>"""


def render_card_start(title: str = "", icon: str = "", padding: str = "20px"):
    """Start a card container (use with markdown)."""
    header = ""
    if title:
        header = f"""
        <div style="
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid {THEME['border']};
        ">
            <span style="font-size: 1.1rem;">{icon}</span>
            <span style="
                color: {THEME['text_primary']};
                font-weight: 600;
                font-size: 0.95rem;
            ">{title}</span>
        </div>"""
    return f"""<div style="
        background: {THEME['bg_surface']};
        border: 1px solid {THEME['border']};
        border-radius: 14px;
        padding: {padding};
    ">{header}"""


def render_card_end():
    """End a card container."""
    return "</div>"


def render_plan(plan: dict, plan_text: str):
    """Display the analysis plan in a premium card."""
    render_section_header("Analysis Plan", "Step-by-step approach", "📋")

    steps_html = ""
    if plan.get("steps"):
        for i, step in enumerate(plan["steps"], 1):
            steps_html += (
                f'<div style="padding:10px 0;border-bottom:1px solid {THEME["border"]}33;">'
                f'<span style="display:inline-block;min-width:26px;height:26px;'
                f'background:{THEME["accent_primary"]}15;border:1px solid {THEME["accent_primary"]}30;'
                f'border-radius:8px;text-align:center;line-height:26px;'
                f'color:{THEME["accent_primary"]};font-size:0.75rem;font-weight:700;'
                f'margin-right:10px;">{i}</span>'
                f'<span style="color:{THEME["text_secondary"]};font-size:0.88rem;">{step}</span></div>'
            )

        badges = ""
        if plan.get("chart_type") and plan["chart_type"] != "none":
            badges += render_status_badge(f"📊 {plan['chart_type'].title()}", "active")
        if plan.get("columns_used"):
            cols_str = ", ".join(plan["columns_used"][:5])
            badges += f" {render_status_badge(f'📁 {cols_str}', 'info')}"
        if badges:
            steps_html += f'<div style="margin-top:12px;">{badges}</div>'
    else:
        steps_html = f'<p style="color:{THEME["text_secondary"]};font-size:0.88rem;">{plan_text}</p>'

    card = (
        f'<div style="background:{THEME["bg_surface"]};border:1px solid {THEME["border"]};'
        f'border-radius:14px;padding:20px;">{steps_html}</div>'
    )
    st.markdown(card, unsafe_allow_html=True)


def render_code(code: str):
    """Display generated code in a premium expander."""
    with st.expander("💻 Generated Code", expanded=False):
        st.code(code, language="python", line_numbers=True)
        st.download_button(
            label="⬇️ Download .py",
            data=code,
            file_name="datummind_analysis.py",
            mime="text/x-python",
            key=f"dl_code_{hash(code)}",
        )


def render_results(execution_result):
    """Display tables and scalar results."""
    if execution_result is None:
        return

    if execution_result.stdout and execution_result.stdout.strip():
        with st.expander("📄 Output Log", expanded=False):
            st.text(execution_result.stdout)

    if execution_result.result_df is not None:
        render_section_header("Results", "Analysis output table", "📊")
        df = execution_result.result_df
        if isinstance(df, pd.DataFrame):
            st.dataframe(df, width="stretch", height=min(400, 50 + len(df) * 35))
        elif isinstance(df, pd.Series):
            st.dataframe(df.to_frame(), width="stretch")
        else:
            st.write(df)

    if execution_result.result_value is not None:
        val = execution_result.result_value
        if isinstance(val, str) and len(val) > 200:
            render_section_header("Model Report", "", "📋")
            st.markdown(val)
        else:
            st.markdown(f"""
            {render_card_start("Result", "📋")}
            <p style="color: {THEME['accent_secondary']}; font-size: 1.3rem; font-weight: 600;">{val}</p>
            {render_card_end()}
            """, unsafe_allow_html=True)


def render_charts(execution_result):
    """Render charts in styled containers."""
    if execution_result is None:
        return

    if execution_result.figures:
        render_section_header("Visualizations", "Auto-generated charts", "📈")
        for i, fig in enumerate(execution_result.figures):
            # Chart container
            st.markdown(f"""
            <div style="
                background: {THEME['bg_surface']};
                border: 1px solid {THEME['border']};
                border-radius: 14px;
                padding: 16px;
                margin-bottom: 12px;
            ">
            """, unsafe_allow_html=True)

            fig.patch.set_facecolor(THEME['bg_surface'])
            for ax in fig.get_axes():
                ax.set_facecolor(THEME['bg_elevated'])
                ax.tick_params(colors=THEME['text_muted'])
                ax.xaxis.label.set_color(THEME['text_secondary'])
                ax.yaxis.label.set_color(THEME['text_secondary'])
                ax.title.set_color(THEME['text_primary'])
                for spine in ax.spines.values():
                    spine.set_color(THEME['border'])
                ax.grid(color=THEME['border'], alpha=0.3)

            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            chart_bytes = save_figure_to_bytes(fig)
            st.download_button(
                label=f"⬇️ Download Chart {i+1}",
                data=chart_bytes,
                file_name=f"datummind_chart_{i+1}.png",
                mime="image/png",
                key=f"dl_chart_{i}_{id(fig)}",
            )

    if hasattr(execution_result, "plotly_figures") and execution_result.plotly_figures:
        for i, fig in enumerate(execution_result.plotly_figures):
            fig.update_layout(
                paper_bgcolor=THEME['bg_surface'],
                plot_bgcolor=THEME['bg_elevated'],
                font_color=THEME['text_secondary'],
            )
            st.plotly_chart(fig, key=f"plotly_{i}_{id(fig)}")


def render_insights(narrative: str):
    """Display insights in a premium card."""
    if not narrative:
        return

    render_section_header("Insights & Analysis", "AI-generated explanation", "💡")
    html = (
        f'<div style="background:{THEME["bg_surface"]};border:1px solid {THEME["border"]};'
        f'border-left:3px solid {THEME["accent_secondary"]};border-radius:14px;padding:22px;">'
        f'<div style="color:{THEME["text_secondary"]};font-size:0.9rem;line-height:1.7;">'
        f'{narrative}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_status_messages(messages: list[str]):
    """Display execution pipeline steps."""
    for msg in messages:
        st.markdown(f"""
        <div style="
            color: {THEME['text_muted']};
            font-size: 0.78rem;
            padding: 3px 0;
            font-weight: 400;
        ">{msg}</div>
        """, unsafe_allow_html=True)


def render_error(error: str):
    """Display a styled error."""
    st.markdown(f"""
    <div style="
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        color: {THEME['error']};
        font-size: 0.88rem;
        font-weight: 500;
    ">❌ {error}</div>
    """, unsafe_allow_html=True)


def render_dataset_info(df: pd.DataFrame, name: str):
    """Show dataset info card in sidebar."""
    rows, cols = df.shape
    nulls = int(df.isnull().sum().sum())
    mem = round(df.memory_usage(deep=True).sum() / 1024, 1)

    badge = render_status_badge("Loaded", "success")
    html = (
        f'<div style="background:{THEME["bg_elevated"]};border:1px solid {THEME["border"]};'
        f'border-radius:10px;padding:12px;margin-bottom:8px;">'
        f'<div style="margin-bottom:8px;">'
        f'<span style="color:{THEME["text_primary"]};font-weight:600;font-size:0.82rem;">📁 {name}</span>'
        f' {badge}</div>'
        f'<div style="color:{THEME["text_muted"]};font-size:0.75rem;">'
        f'📐 {rows:,} × {cols} &nbsp;&nbsp; 💾 {mem} KB &nbsp;&nbsp; ⚠️ {nulls} nulls'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("Preview", expanded=False):
        st.dataframe(df.head(8), width="stretch", height=200)


def render_kpi_row(df: pd.DataFrame):
    """Render a row of KPI cards from dataset summary."""
    cols = st.columns(4)
    rows, num_cols = df.shape

    with cols[0]:
        render_metric_card("Total Rows", f"{rows:,}", icon="📐", color=THEME['accent_primary'])
    with cols[1]:
        render_metric_card("Columns", str(num_cols), icon="📊", color=THEME['accent_secondary'])
    with cols[2]:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        render_metric_card("Numeric", str(numeric_cols), icon="🔢", color=THEME['success'])
    with cols[3]:
        null_pct = round(df.isnull().mean().mean() * 100, 1)
        color = THEME['success'] if null_pct < 5 else THEME['warning'] if null_pct < 20 else THEME['error']
        render_metric_card("Missing %", f"{null_pct}%", icon="⚠️", color=color)


def render_column_stats(df: pd.DataFrame):
    """Render column statistics in a premium table."""
    stats = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            type_badge = "🔢 Numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            type_badge = "📅 DateTime"
        else:
            type_badge = "🏷️ Category"
        stats.append({
            "Column": col,
            "Type": type_badge,
            "Unique": unique,
            "Nulls": nulls,
            "Dtype": dtype,
        })

    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df, width="stretch", hide_index=True, height=min(400, 40 + len(stats) * 35))


def render_query_card(icon: str, query: str, key: str) -> bool:
    """Render a sample query as a clickable card. Returns True if clicked."""
    return st.button(
        f"{icon}  {query}",
        key=key,
        use_container_width=True,
    )


def render_sidebar_divider():
    """Render a subtle sidebar divider."""
    st.markdown(f"""
    <div style="
        height: 1px;
        background: {THEME['border']};
        margin: 16px 0;
        opacity: 0.5;
    "></div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the dashboard footer."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 32px 0 16px 0;
        color: {THEME['text_muted']};
        font-size: 0.75rem;
        border-top: 1px solid {THEME['border']};
        margin-top: 40px;
    ">
        🧠 <strong style="color: {THEME['text_secondary']}">DatumMind Analyst</strong> v2.0 &nbsp;·&nbsp;
        AI-Powered Data Analysis &nbsp;·&nbsp;
        Built with Streamlit, Groq, & Python
    </div>
    """, unsafe_allow_html=True)
