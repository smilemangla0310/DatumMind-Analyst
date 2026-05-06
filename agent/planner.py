"""
Analysis plan generation via LLM.
Converts a natural language question into a step-by-step analysis plan.
"""

PLANNER_SYSTEM_PROMPT = """You are DatumMind Analyst's Planning Engine. Your job is to create a clear, step-by-step analysis plan for a data analysis question.

Given a user question and dataset schema, produce a structured plan with numbered steps.

Rules:
1. Each step should be a concrete action: filter, group, aggregate, compute, plot, etc.
2. Reference specific column names from the schema.
3. If the question is ambiguous, make reasonable assumptions and note them.
4. If a column doesn't exist in the schema, say so clearly.
5. Choose appropriate chart types:
   - Comparisons across categories → bar chart
   - Trends over time → line chart
   - Relationships between two numeric variables → scatter plot
   - Distribution of a single variable → histogram or box plot
   - Correlation matrix → heatmap
   - Composition/proportion → pie chart or stacked bar
6. For ML tasks, include: data preparation, model selection, training, evaluation steps.
7. Keep the plan concise (5-10 steps max).

Respond ONLY with the plan in this format:
PLAN:
1. [Step description]
2. [Step description]
...
CHART_TYPE: [bar|line|scatter|histogram|box|heatmap|pie|none]
COLUMNS_USED: [comma-separated list of column names]
ASSUMPTIONS: [any assumptions made, or "None"]
"""


def build_planner_prompt(query: str, schema_str: str, conversation_context: str = "") -> str:
    """Build the full prompt for the planner."""
    parts = [
        f"## Dataset Schema\n{schema_str}",
    ]

    if conversation_context and conversation_context != "No previous conversation.":
        parts.append(f"\n## Previous Conversation Context\n{conversation_context}")

    parts.append(f"\n## User Question\n{query}")

    return "\n".join(parts)


def parse_plan_response(response_text: str) -> dict:
    """Parse the LLM's plan response into structured data."""
    result = {
        "raw_plan": response_text,
        "steps": [],
        "chart_type": "none",
        "columns_used": [],
        "assumptions": "None",
    }

    lines = response_text.strip().split("\n")
    current_section = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if line_stripped.upper().startswith("PLAN:"):
            current_section = "plan"
            continue
        elif line_stripped.upper().startswith("CHART_TYPE:"):
            result["chart_type"] = line_stripped.split(":", 1)[1].strip().lower()
            current_section = None
            continue
        elif line_stripped.upper().startswith("COLUMNS_USED:"):
            cols = line_stripped.split(":", 1)[1].strip()
            result["columns_used"] = [c.strip() for c in cols.split(",") if c.strip()]
            current_section = None
            continue
        elif line_stripped.upper().startswith("ASSUMPTIONS:"):
            result["assumptions"] = line_stripped.split(":", 1)[1].strip()
            current_section = None
            continue

        if current_section == "plan":
            # Extract step text (remove numbering)
            step = line_stripped.lstrip("0123456789.-) ").strip()
            if step:
                result["steps"].append(step)

    # If no structured plan was found, treat the whole response as the plan
    if not result["steps"]:
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.upper().startswith(("CHART_TYPE", "COLUMNS_USED", "ASSUMPTIONS")):
                result["steps"].append(line_stripped)

    return result
