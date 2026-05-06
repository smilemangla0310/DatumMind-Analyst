"""
App configuration, theme constants, and design tokens.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── App Settings ──────────────────────────────────────────────────────────────
APP_TITLE = "DatumMind Analyst"
APP_ICON = "🧠"
APP_SUBTITLE = "AI-Powered Data Analysis Agent"
APP_DESCRIPTION = "Upload your data, ask questions in natural language, and get instant analysis with visualizations and insights."
APP_VERSION = "2.0"

# ── File Settings ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 200
SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls"]

# ── Design Tokens ─────────────────────────────────────────────────────────────
THEME = {
    # Backgrounds
    "bg_primary": "#0F1117",
    "bg_surface": "#1A1D29",
    "bg_surface_hover": "#222638",
    "bg_elevated": "#252836",
    # Borders
    "border": "#2A2D3E",
    "border_accent": "#3B3F54",
    # Accent Colors
    "accent_primary": "#6366F1",       # Indigo
    "accent_secondary": "#22D3EE",     # Cyan
    "accent_tertiary": "#8B5CF6",      # Violet
    "accent_gradient": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%)",
    # Status
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
    # Text
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "text_accent": "#A5B4FC",
    # Glow
    "glow_primary": "0 0 20px rgba(99, 102, 241, 0.15)",
    "glow_accent": "0 0 15px rgba(34, 211, 238, 0.10)",
}

# ── LLM Provider Settings ────────────────────────────────────────────────────
LLM_PROVIDERS = {
    "Groq (Free)": {
        "key": "groq",
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "help": "Free key → https://console.groq.com/keys (14,400 req/day)",
    },
    "HuggingFace (Free)": {
        "key": "huggingface",
        "env_var": "HF_API_KEY",
        "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "models": [
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        "help": "Free token → https://huggingface.co/settings/tokens",
    },
    "Google Gemini": {
        "key": "gemini",
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ],
        "help": "Free key → https://aistudio.google.com/apikey",
    },
}

DEFAULT_PROVIDER = "Groq (Free)"
TEMPERATURE = 0.1
MAX_RETRIES = 3

# ── Env keys (auto-load from .env) ───────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Sample Queries ────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    ("📊", "What are the total revenue and units sold by region?"),
    ("📈", "Show me the monthly revenue trend over time"),
    ("💰", "Which product has the highest profit margin?"),
    ("🔗", "What's the correlation between customer age and satisfaction?"),
    ("📡", "Compare revenue performance across sales channels"),
    ("🤖", "Build a regression model to predict revenue"),
    ("🎯", "Cluster customers into segments based on behavior"),
    ("🔍", "Detect anomalies in the revenue data"),
]

# ── Streamlit Page Config ─────────────────────────────────────────────────────
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
