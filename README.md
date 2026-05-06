# 🧠 DatumMind Analyst

**Premium AI-Powered Data Analysis Dashboard** — Upload datasets, ask questions in natural language, and get instant analysis with visualizations and business insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange?logo=meta)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 🎨 Premium Dark-Theme Dashboard
Modern SaaS-grade analytics workspace with layered dark surfaces, KPI metric cards, status badges, tabbed navigation, and polished visual hierarchy.

### 🤖 Multi-Provider LLM Support
Choose from three AI providers:
- **Groq** (Recommended) — 14,400 free requests/day with LLaMA 3.3 70B
- **HuggingFace** — Free tier with Qwen, Mistral, LLaMA models
- **Google Gemini** — Gemini 2.0 Flash

### 🔍 Natural Language Analysis
Ask complex business questions in plain English — DatumMind converts them into executable Python code automatically.

### 📋 Transparent Analysis Plans
See the step-by-step plan before execution: what columns to inspect, what filters to apply, which charts to generate.

### 💻 Auto-Generated Code
View the pandas/matplotlib/seaborn/sklearn code that powers every analysis. Download it as a `.py` file for reuse.

### 📊 Smart Visualizations
Dark-themed charts that match the dashboard. Supports bar, line, scatter, histogram, box plot, heatmap, and interactive Plotly charts.

### 💡 Business Insights
Every analysis comes with a plain-English narrative: key findings, patterns, outliers, and actionable business insights.

### 🤖 ML Capabilities
- **Regression**: Predict continuous values
- **Clustering**: Segment data with KMeans
- **Anomaly Detection**: Find outliers with IsolationForest
- **Classification**: Categorize with RandomForest

### 🧠 Conversation Memory
Ask follow-up questions like "Now break this down by region" — the agent remembers context.

### 🔍 Interactive Data Explorer
Browse, filter, search, and visualize any column — all without writing code.

---

## 🏗️ Architecture

```
DatumMind-Analyst/
├── app/                        # Streamlit UI layer
│   ├── main.py                 # Dashboard entry point
│   ├── config.py               # Theme tokens, provider config
│   └── ui_components.py        # Premium card/metric components
├── agent/                      # AI agent layer
│   ├── core.py                 # Multi-provider orchestrator
│   ├── planner.py              # Analysis plan generation
│   ├── code_generator.py       # Code + narrative prompts
│   ├── executor.py             # Sandboxed execution engine
│   └── memory.py               # Conversation memory
├── analysis/                   # Data & ML utilities
│   ├── data_utils.py           # Data loading, cleaning, schema
│   ├── viz_utils.py            # Dark-theme chart styling
│   └── ml_engine.py            # ML evaluation & reporting
├── sample_data/
│   └── sales_sample.csv        # Demo dataset (200+ rows)
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── requirements.txt
├── .env.example
└── README.md
```

### Agent Pipeline

```
User Question
     ↓
┌─────────────┐
│   Planner   │ → Step-by-step analysis plan
└──────┬──────┘
       ↓
┌─────────────┐
│  Code Gen   │ → pandas/matplotlib/sklearn code
└──────┬──────┘
       ↓
┌─────────────┐
│  Executor   │ → Sandboxed exec() with retry
└──────┬──────┘
       ↓
┌─────────────┐
│  Narrator   │ → Business insights
└──────┬──────┘
       ↓
   Dashboard (plan + code + tables + charts + narrative)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An API key from one of: [Groq](https://console.groq.com/keys), [HuggingFace](https://huggingface.co/settings/tokens), or [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/smilemangla0310/DatumMind-Analyst.git
cd DatumMind-Analyst

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the env template
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and add your API key(s)
```

### Run

```bash
streamlit run app/main.py
```

The dashboard opens at `http://localhost:8501`.

---

## 📝 Example Queries

### Data Analysis
- "What are the total revenue and units sold by region?"
- "Show me the monthly revenue trend over time"
- "Which product has the highest profit margin?"
- "Compare revenue performance across sales channels"

### Machine Learning
- "Build a regression model to predict revenue from units sold and cost"
- "Cluster customers into segments based on revenue and units"
- "Detect anomalies in the revenue data"

### Follow-Ups
- "Now break this down by product"
- "Filter to only the North region"
- "Show this as a pie chart"

---

## 🌐 Deployment (Streamlit Community Cloud)

1. **Push to GitHub**
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: repo → branch `main` → file `app/main.py`
4. **Add Secrets**:
   ```toml
   GROQ_API_KEY = "your_key"
   ```
5. Live at `https://your-app.streamlit.app` 🎉

---

## 🛡️ Safety

| Feature | Description |
|---|---|
| Sandboxed Execution | Restricted `exec()` with blocked builtins |
| Code Validation | Pre-scan for dangerous patterns |
| 30s Timeout | Prevents infinite loops |
| Self-Healing | Auto-fixes failed code (up to 3 retries) |
| Transparency | All generated code shown to user |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit (dark theme) |
| LLM | Groq / HuggingFace / Google Gemini |
| Data | pandas, NumPy |
| Charts | matplotlib, seaborn, Plotly |
| ML | scikit-learn |
| Fonts | Inter (Google Fonts) |

---

## 📄 License

MIT License

---

<p align="center">
  Built with ❤️ using Streamlit & AI
</p>
