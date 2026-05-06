"""
Data loading, cleaning, and schema inspection utilities.
"""

import pandas as pd
import numpy as np
import io
from typing import Any


def load_dataset(file) -> pd.DataFrame:
    """Load a dataset from an uploaded file (CSV or Excel)."""
    filename = getattr(file, "name", "unknown.csv").lower()

    try:
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            # Try reading as CSV with various encodings
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Failed to load file '{filename}': {str(e)}")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    return df


def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic automatic data cleaning."""
    df = df.copy()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]

    # Try to parse date-like columns
    for col in df.select_dtypes(include=["object"]).columns:
        if any(kw in col.lower() for kw in ["date", "time", "timestamp", "created", "updated"]):
            try:
                df[col] = pd.to_datetime(df[col], format="mixed")
            except (ValueError, TypeError):
                pass

    # Downcast numeric columns to save memory
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def get_schema(df: pd.DataFrame) -> dict:
    """Extract a comprehensive schema from a DataFrame."""
    schema = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    for col in df.columns:
        col_info: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 1),
            "unique_count": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
            col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            col_info["mean"] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["min"] = str(df[col].min())
            col_info["max"] = str(df[col].max())
        else:
            # Categorical / string
            top_values = df[col].value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        col_info["sample_values"] = [str(v) for v in df[col].dropna().head(3).tolist()]
        schema["columns"][col] = col_info

    return schema


def format_schema_for_llm(schema: dict) -> str:
    """Format schema dict into a human-readable string for LLM prompts."""
    lines = []
    lines.append(f"Dataset: {schema['shape']['rows']} rows × {schema['shape']['columns']} columns")
    lines.append(f"Memory: {schema['memory_usage_mb']} MB")
    lines.append("")
    lines.append("Columns:")
    lines.append("-" * 60)

    for col_name, info in schema["columns"].items():
        dtype = info["dtype"]
        nulls = info["null_count"]
        unique = info["unique_count"]
        line = f"  • {col_name} ({dtype}) — {unique} unique, {nulls} nulls ({info['null_pct']}%)"

        if "min" in info and info["min"] is not None:
            line += f", range: [{info['min']} → {info['max']}]"
            if "mean" in info:
                line += f", mean: {info['mean']}"
        elif "top_values" in info:
            top = list(info["top_values"].keys())[:3]
            line += f", top values: {top}"

        lines.append(line)

    return "\n".join(lines)


def get_summary_stats(df: pd.DataFrame) -> str:
    """Generate a descriptive statistics summary string."""
    buf = io.StringIO()
    df.describe(include="all").to_string(buf=buf)
    return buf.getvalue()
