"""Structured AI decision schema (Phase 5).

Defines the canonical shape of a *suggestion* the model can produce: a single
:class:`Action` (Phase-3 frozen dataclass), optional reasoning, and
confidence. The model does not invent new field names or action kinds -- it
is constrained to the existing Phase-3 action vocabulary and the Phase-4
validation rules.

This is intentionally minimal: Phase 5 ships a *decision* (one action +
metadata). Phase 6 may extend to multi-step plans, but the decision unit
stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions.action import Action, ActionValidationError
from .base import AIError


@dataclass(frozen=True)
class AIDecision:
    """A single AI-generated action suggestion.

    The ``action`` field is the only field with semantic content; the rest
    are metadata for logging, auditing, and safety gating.

    Invariant:
        :attr:`action` is always a valid :class:`Action` that has passed
        Phase-3 field validation. :meth:`from_dict` enforces this and raises
        :class:`AIError` on any malformed payload.
    """

    action: Action
    #: Human-readable rationale from the model (optional, for audit/log).
    reasoning: str = ""
    #: Model-reported confidence in [0.0, 1.0]; informational only.
    #: The application may apply its own confidence gate.
    confidence: float = 1.0

    #: Schema version for forward compatibility.
    SCHEMA_VERSION: int = 1

    def __post_init__(self) -> None:
        # Re-validate the Action to catch any bypass of from_dict.
        from ..actions.action import validate_action

        try:
            validate_action(self.action)
        except ActionValidationError as exc:
            raise AIError(
                f"AI suggested an invalid action: {exc}"
            ) from exc
        if not 0.0 <= self.confidence <= 1.0:
            raise AIError(
                f"AI decision confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )

    @property
    def kind(self) -> str:
        """Shorthand for the action kind (e.g. ``"click"``)."""
        return self.action.kind.value

    def to_dict(self) -> dict:
        """Serialize for logging / API transport."""
        return {
            "action": self.action.to_dict(),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AIDecision:
        """Deserialize and validate an AI decision payload.

        Expected shape::

            {
                "action": { ...Action dict... },
                "reasoning": "...",       # optional
                "confidence": 0.92,       # optional, default 1.0
            }

        Args:
            data: Raw dict (typically from ``json.loads`` of model output).

        Returns:
            A validated :class:`AIDecision`.

        Raises:
            AIError: On any missing/invalid field or action violation.
        """
        if not isinstance(data, dict):
            raise AIError(
                f"AI decision must be a JSON object, got {type(data).__name__}"
            )

        raw_action = data.get("action")
        if raw_action is None:
            raise AIError("AI decision missing 'action' field")
        if not isinstance(raw_action, dict):
            raise AIError(
                f"AI decision 'action' must be an object, got {type(raw_action).__name__}"
            )

        try:
            action = Action.from_dict(raw_action)
        except ActionValidationError as exc:
            raise AIError(f"AI suggested an invalid action: {exc}") from exc

        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            raise AIError(
                f"AI decision 'reasoning' must be a string, got {type(reasoning).__name__}"
            )

        confidence = data.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise AIError(
                f"AI decision 'confidence' must be a number, got {type(confidence).__name__}"
            )

        return cls(action=action, reasoning=reasoning, confidence=float(confidence))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AIDecision(kind={self.kind!r}, "
            f"confidence={self.confidence:.3f}, "
            f"reasoning={self.reasoning[:40]!r}...)"
        )
