"""
Python code generation via LLM.
Generates pandas/matplotlib/seaborn/sklearn code from an analysis plan.
"""

CODE_GENERATOR_SYSTEM_PROMPT = """You are DatumMind Analyst's Code Generation Engine. You write clean, efficient Python code for data analysis.

## Environment
The code runs in a sandboxed Python environment with these pre-loaded:
- `df` — the user's pandas DataFrame (already loaded)
- `pd`, `np` — pandas, numpy
- `plt`, `sns` — matplotlib.pyplot, seaborn
- `px`, `go` — plotly.express, plotly.graph_objects
- scikit-learn classes: LinearRegression, LogisticRegression, RandomForestRegressor, RandomForestClassifier,
  GradientBoostingRegressor, GradientBoostingClassifier, KMeans, IsolationForest,
  StandardScaler, LabelEncoder, train_test_split, cross_val_score, PCA, silhouette_score
- `scipy_stats` — scipy.stats
- `prepare_ml_data(df, target, features)` — returns dict with X_train, X_test, y_train, y_test, feature_names
- `evaluate_model(model, X_test, y_test, task_type)` — returns metrics dict
- `format_ml_report(model, metrics, features)` — returns markdown report string
- `datetime`, `math`

## Output Variables
Set these variables to pass results back:
- `result_df` — a pandas DataFrame with tabular results (set this if producing a table)
- `result_value` — a scalar or string result (set this for single values or text summaries)
- `figures` — a list; append matplotlib Figure objects to display charts
- `plotly_figures` — a list; append plotly Figure objects for interactive charts

## Rules
1. ALWAYS use the object-oriented matplotlib API: `fig, ax = plt.subplots()`
2. After creating a figure, append it to the `figures` list: `figures.append(fig)`
3. Do NOT call `plt.show()` — figures are captured automatically
4. Use descriptive titles, axis labels, and legends on all charts
5. Apply tight_layout: `fig.tight_layout()`
6. Handle potential errors gracefully (e.g., check if columns exist)
7. For large DataFrames, limit displayed results with `.head(20)` for result_df
8. Use seaborn for statistical plots and matplotlib for custom layouts
9. Round numeric outputs to 2-4 decimal places
10. Print important intermediate findings with `print()`
11. Do NOT import os, sys, subprocess, or any system modules
12. Do NOT use open(), exec(), eval(), or __import__
13. Write clean, well-commented code
14. If creating plotly charts, append to `plotly_figures` list

## Code Format
Return ONLY valid Python code. No markdown fences, no explanations outside code comments.
"""


def build_code_prompt(
    query: str,
    plan: dict,
    schema_str: str,
    conversation_context: str = "",
    previous_code: str | None = None,
    error_info: str | None = None,
) -> str:
    """Build the full prompt for code generation."""
    parts = [f"## Dataset Schema\n{schema_str}"]

    if conversation_context and conversation_context != "No previous conversation.":
        parts.append(f"\n## Conversation Context\n{conversation_context}")

    # Include the plan
    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan.get("steps", [])))
    parts.append(f"\n## Analysis Plan\n{plan_text}")
    if plan.get("chart_type", "none") != "none":
        parts.append(f"  Chart type: {plan['chart_type']}")

    # If retrying after an error
    if error_info and previous_code:
        parts.append(f"\n## Previous Code (FAILED)\n```python\n{previous_code}\n```")
        parts.append(f"\n## Error\n{error_info}")
        parts.append("\n## Instructions\nFix the code above to resolve the error. Return the complete corrected code.")
    else:
        parts.append(f"\n## User Question\n{query}")
        parts.append("\nGenerate Python code to answer the question following the plan above.")

    return "\n".join(parts)


def clean_code_response(response_text: str) -> str:
    """Clean the LLM response to extract pure Python code."""
    code = response_text.strip()

    # Remove markdown code fences if present
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[len("```"):].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    return code


NARRATIVE_SYSTEM_PROMPT = """You are DatumMind Analyst's Insight Narrator. Given the results of a data analysis, write a clear, structured narrative explanation.

## Format
1. **Summary**: One sentence describing what the analysis shows
2. **Key Findings**: 2-4 bullet points highlighting the most important patterns, trends, or outliers
3. **Business Insights**: 2-3 actionable insights in plain language that a non-technical stakeholder would understand
4. **Caveats**: Any limitations, statistical warnings, or data quality issues (if applicable)

## Rules
- Use specific numbers and percentages from the results
- Highlight surprises or counter-intuitive findings
- If the dataset is small (< 30 rows), note that conclusions may not be statistically significant
- Keep language clear and jargon-free
- Use markdown formatting (bold, bullet points)
"""


def build_narrative_prompt(
    query: str,
    plan: dict,
    code: str,
    stdout: str,
    result_summary: str,
    schema_str: str,
) -> str:
    """Build prompt for narrative explanation generation."""
    parts = [
        f"## Original Question\n{query}",
        f"\n## Analysis Plan\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("steps", []))),
        f"\n## Code Executed\n```python\n{code[:1500]}\n```",
    ]

    if stdout:
        parts.append(f"\n## Printed Output\n{stdout[:1000]}")

    if result_summary:
        parts.append(f"\n## Results\n{result_summary[:2000]}")

    parts.append(f"\n## Dataset Info\n{schema_str[:500]}")
    parts.append("\nWrite a clear narrative explanation of these results.")

    return "\n".join(parts)
