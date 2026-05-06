"""
DatumMind Agent Core — main orchestrator.
Pipeline: query → plan → code → execute → debug → narrate
Supports multiple LLM providers: Groq, HuggingFace, Google Gemini.
"""

import os
import io
import pandas as pd
from typing import Any, Callable
from dataclasses import dataclass, field

from agent.memory import ConversationMemory
from agent.planner import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
    parse_plan_response,
)
from agent.code_generator import (
    CODE_GENERATOR_SYSTEM_PROMPT,
    NARRATIVE_SYSTEM_PROMPT,
    build_code_prompt,
    clean_code_response,
    build_narrative_prompt,
)
from agent.executor import execute_code, ExecutionResult
from analysis.data_utils import get_schema, format_schema_for_llm


@dataclass
class AnalysisResponse:
    """Complete response from the agent."""
    plan: dict = field(default_factory=dict)
    plan_text: str = ""
    code: str = ""
    execution: ExecutionResult | None = None
    narrative: str = ""
    status_messages: list[str] = field(default_factory=list)
    error: str = ""
    attempts: int = 0


# ── LLM Provider Implementations ─────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    def call(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq API provider — fast, generous free tier."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "invalid" in error_str:
                raise RuntimeError(
                    "🔑 Invalid Groq API Key. Get a free key from https://console.groq.com/keys"
                )
            elif "rate" in error_str or "429" in error_str:
                raise RuntimeError(
                    "⏳ Groq rate limit reached. Please wait a moment and try again."
                )
            else:
                raise RuntimeError(f"Groq API error: {str(e)}")


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider."""

    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(temperature, 0.01),  # HF doesn't accept 0
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str or "token" in error_str:
                raise RuntimeError(
                    "🔑 Invalid HuggingFace Token. Get a free token from https://huggingface.co/settings/tokens"
                )
            elif "rate" in error_str or "429" in error_str:
                raise RuntimeError(
                    "⏳ HuggingFace rate limit reached. Please wait a moment and try again."
                )
            else:
                raise RuntimeError(f"HuggingFace API error: {str(e)}")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._types = __import__("google.genai.types", fromlist=["types"])

    def call(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        from google.genai import types
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
        except Exception as e:
            error_str = str(e).lower()
            if "api key not valid" in error_str or "invalid_argument" in error_str:
                raise RuntimeError(
                    "🔑 Invalid Google API Key. Get a free key from https://aistudio.google.com/apikey"
                )
            elif "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                raise RuntimeError(
                    "⏳ Gemini quota exhausted. Try switching to Groq (free, generous limits) in the sidebar."
                )
            else:
                raise RuntimeError(f"Gemini API error: {str(e)}")


def create_provider(provider_key: str, api_key: str, model: str) -> LLMProvider:
    """Factory function to create the right LLM provider."""
    if provider_key == "groq":
        return GroqProvider(api_key=api_key, model=model)
    elif provider_key == "huggingface":
        return HuggingFaceProvider(api_key=api_key, model=model)
    elif provider_key == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_key}")


# ── Main Agent ────────────────────────────────────────────────────────────────

class DatumMindAgent:
    """
    Main AI agent orchestrator.
    Manages the full pipeline: understand → plan → code → execute → debug → explain.
    """

    def __init__(
        self,
        provider_key: str = "groq",
        api_key: str = "",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.temperature = temperature
        self.max_retries = max_retries
        self.memory = ConversationMemory()
        self.provider_key = provider_key
        self.model_name = model_name

        if not api_key:
            raise ValueError("API key is required. Please enter one in the sidebar.")

        self.provider = create_provider(provider_key, api_key, model_name)

        # Dataframes storage
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.schemas: dict[str, dict] = {}
        self.schema_strings: dict[str, str] = {}

    def load_dataset(self, name: str, df: pd.DataFrame):
        """Register a dataset with the agent."""
        self.dataframes[name] = df
        schema = get_schema(df)
        self.schemas[name] = schema
        self.schema_strings[name] = format_schema_for_llm(schema)

    def get_combined_schema_string(self) -> str:
        """Get schema strings for all loaded datasets."""
        if len(self.schema_strings) == 1:
            return list(self.schema_strings.values())[0]

        parts = []
        for name, schema_str in self.schema_strings.items():
            parts.append(f"### Dataset: {name}\n{schema_str}")
        return "\n\n".join(parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to the configured LLM provider."""
        return self.provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=4096,
        )

    def _format_result_summary(self, result: ExecutionResult) -> str:
        """Create a text summary of execution results for the LLM."""
        parts = []

        if result.stdout:
            parts.append(f"Stdout:\n{result.stdout[:1000]}")

        if result.result_df is not None:
            buf = io.StringIO()
            result.result_df.head(20).to_string(buf=buf)
            parts.append(f"Result Table:\n{buf.getvalue()}")

        if result.result_value is not None:
            parts.append(f"Result Value: {str(result.result_value)[:500]}")

        if result.figures:
            parts.append(f"Charts generated: {len(result.figures)} figure(s)")

        if hasattr(result, "plotly_figures") and result.plotly_figures:
            parts.append(f"Interactive charts: {len(result.plotly_figures)} plotly figure(s)")

        return "\n".join(parts) if parts else "No output captured."

    def analyze(
        self,
        query: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> AnalysisResponse:
        """
        Run the full analysis pipeline for a user query.

        Args:
            query: Natural language question from the user
            status_callback: Optional function to report progress steps

        Returns:
            AnalysisResponse with plan, code, results, and narrative
        """
        response = AnalysisResponse()

        def emit(msg: str):
            response.status_messages.append(msg)
            if status_callback:
                status_callback(msg)

        try:
            # ── Pre-checks ──────────────────────────────────────────────
            if not self.dataframes:
                response.error = "No dataset loaded. Please upload a CSV or Excel file first."
                return response

            schema_str = self.get_combined_schema_string()
            context = self.memory.get_context()

            # ── Step 1: Generate Analysis Plan ──────────────────────────
            emit("📋 Generating analysis plan...")

            planner_prompt = build_planner_prompt(query, schema_str, context)
            plan_response = self._call_llm(PLANNER_SYSTEM_PROMPT, planner_prompt)
            plan = parse_plan_response(plan_response)
            response.plan = plan
            response.plan_text = plan_response

            if not plan["steps"]:
                response.error = "Could not generate a valid analysis plan for this question."
                return response

            emit(f"✅ Plan ready — {len(plan['steps'])} steps")

            # ── Step 2: Generate Code ───────────────────────────────────
            emit("💻 Writing Python code...")

            code_prompt = build_code_prompt(
                query=query,
                plan=plan,
                schema_str=schema_str,
                conversation_context=context,
            )
            code_response = self._call_llm(CODE_GENERATOR_SYSTEM_PROMPT, code_prompt)
            code = clean_code_response(code_response)
            response.code = code

            emit("✅ Code generated")

            # ── Step 3: Execute with retry loop ─────────────────────────
            exec_result = None
            current_code = code

            for attempt in range(1, self.max_retries + 1):
                response.attempts = attempt
                emit(f"⚙️ Executing code (attempt {attempt}/{self.max_retries})...")

                exec_result = execute_code(current_code, self.dataframes)

                if exec_result.success:
                    emit("✅ Code executed successfully")
                    break
                else:
                    emit(f"⚠️ Error: {exec_result.error[:100]}")

                    if attempt < self.max_retries:
                        emit("🔧 Analyzing error and fixing code...")

                        # Ask LLM to fix the code
                        fix_prompt = build_code_prompt(
                            query=query,
                            plan=plan,
                            schema_str=schema_str,
                            conversation_context=context,
                            previous_code=current_code,
                            error_info=f"{exec_result.error}\n{exec_result.error_traceback}",
                        )
                        fix_response = self._call_llm(CODE_GENERATOR_SYSTEM_PROMPT, fix_prompt)
                        current_code = clean_code_response(fix_response)
                        response.code = current_code  # Update to latest code
                    else:
                        emit("❌ Code execution failed after all attempts")

            response.execution = exec_result

            if not exec_result or not exec_result.success:
                error_msg = exec_result.error if exec_result else "Unknown execution error"
                response.error = f"Analysis failed: {error_msg}"
                # Still try to generate a narrative about the failure
                self.memory.add_user_turn(query)
                self.memory.add_assistant_turn(
                    content=f"Failed to analyze: {error_msg}",
                    plan=response.plan_text,
                    code=response.code,
                    error=error_msg,
                )
                return response

            # ── Step 4: Generate Narrative ──────────────────────────────
            emit("📝 Generating insights narrative...")

            result_summary = self._format_result_summary(exec_result)
            narrative_prompt = build_narrative_prompt(
                query=query,
                plan=plan,
                code=current_code,
                stdout=exec_result.stdout,
                result_summary=result_summary,
                schema_str=schema_str,
            )
            narrative = self._call_llm(NARRATIVE_SYSTEM_PROMPT, narrative_prompt)
            response.narrative = narrative

            emit("✅ Analysis complete!")

            # ── Step 5: Update Memory ───────────────────────────────────
            self.memory.add_user_turn(query)
            self.memory.add_assistant_turn(
                content=narrative[:500],
                plan=response.plan_text[:300],
                code=current_code[:300],
                results_summary=result_summary[:300],
            )

        except Exception as e:
            response.error = f"Agent error: {str(e)}"
            emit(f"❌ Error: {str(e)}")

        return response

    def clear_memory(self):
        """Reset conversation memory."""
        self.memory.clear()

    def clear_datasets(self):
        """Remove all loaded datasets."""
        self.dataframes.clear()
        self.schemas.clear()
        self.schema_strings.clear()
