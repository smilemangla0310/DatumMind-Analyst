"""
Conversation memory management for follow-up questions.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    plan: Optional[str] = None
    code: Optional[str] = None
    results_summary: Optional[str] = None
    error: Optional[str] = None


class ConversationMemory:
    """Manages conversation history for multi-turn interactions."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: list[ConversationTurn] = []
        self.dataset_names: list[str] = []

    def add_user_turn(self, content: str):
        """Add a user message."""
        self.turns.append(ConversationTurn(role="user", content=content))
        self._trim()

    def add_assistant_turn(
        self,
        content: str,
        plan: str | None = None,
        code: str | None = None,
        results_summary: str | None = None,
        error: str | None = None,
    ):
        """Add an assistant response with optional analysis artifacts."""
        self.turns.append(
            ConversationTurn(
                role="assistant",
                content=content,
                plan=plan,
                code=code,
                results_summary=results_summary,
                error=error,
            )
        )
        self._trim()

    def get_context(self, max_turns: int | None = None) -> str:
        """Get formatted conversation history for LLM context."""
        limit = max_turns or self.max_turns
        recent = self.turns[-limit:]
        if not recent:
            return "No previous conversation."

        lines = ["=== Conversation History ==="]
        for turn in recent:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"\n[{role}]: {turn.content}")

            if turn.plan:
                lines.append(f"  [Plan]: {turn.plan[:300]}...")
            if turn.code:
                lines.append(f"  [Code]: {turn.code[:200]}...")
            if turn.results_summary:
                lines.append(f"  [Result]: {turn.results_summary[:300]}...")
            if turn.error:
                lines.append(f"  [Error]: {turn.error[:200]}")

        return "\n".join(lines)

    def get_last_code(self) -> str | None:
        """Get the most recent generated code."""
        for turn in reversed(self.turns):
            if turn.code:
                return turn.code
        return None

    def get_last_results(self) -> str | None:
        """Get the most recent results summary."""
        for turn in reversed(self.turns):
            if turn.results_summary:
                return turn.results_summary
        return None

    def clear(self):
        """Clear all conversation history."""
        self.turns.clear()

    def _trim(self):
        """Trim history to max_turns."""
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def __len__(self) -> int:
        return len(self.turns)
