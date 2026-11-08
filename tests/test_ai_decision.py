"""AIDecision data-model tests.

Split from tests/test_ai.py (2026-11-08).

Verifies from_dict/to_dict round-trips, validation (reasoning type,
confidence range and type), provenance, and the helper Action builders.
Pure (no network).
"""

from __future__ import annotations

import json

import pytest


def _click_action():
    """A valid minimal mouse_click Action (Phase-3 canonical vocabulary)."""
    from ai_game_agent.actions import Action

    return Action.from_dict(_noop_action_dict())


class TestAIDecisionDirectConstruction:
    """Direct construction must apply the same invariants as from_dict.

    bool is an int subclass in Python, so range checks alone would accept
    confidence=True; from_dict rejects both bool confidence and non-string
    reasoning, and the dataclass constructor must match those rules.
    """

    def test_direct_valid_decision(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision(action=_click_action(), reasoning="why", confidence=0.4)
        assert dec.confidence == 0.4
        assert dec.reasoning == "why"

    def test_direct_non_string_reasoning_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision(action=_click_action(), reasoning=42)  # type: ignore[arg-type]

    def test_direct_bool_confidence_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision(action=_click_action(), confidence=True)  # type: ignore[arg-type]

    def test_direct_nan_confidence_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision(action=_click_action(), confidence=float("nan"))

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
    def test_direct_infinite_confidence_rejected(self, bad: float):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision(action=_click_action(), confidence=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.0001, -1.0, 2.0])
    def test_direct_out_of_range_confidence_rejected(self, bad: float):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision(action=_click_action(), confidence=bad)

    @pytest.mark.parametrize("edge", [0.0, 1.0])
    def test_direct_boundary_confidence_accepted(self, edge: float):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision(action=_click_action(), confidence=edge)
        assert dec.confidence == edge

    def test_from_dict_nan_confidence_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": float("nan")}
            )

    def test_from_dict_infinite_confidence_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": float("inf")}
            )

    def test_from_dict_non_string_reasoning_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "reasoning": ["a", "b"]}
            )


class TestAIDecisionProvenance:
    """An AI-generated decision's action is AI-originated.

    ``Action.source`` defaults to ``"user"``. An :class:`AIDecision` must
    stamp its action's provenance to ``"ai"`` so the executor, logs, and
    replay can distinguish AI actions from human-entered ones.
    """

    def test_from_dict_stamps_ai_source(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        assert dec.action.source == "ai"

    def test_direct_construction_stamps_ai_source(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision(action=_click_action(), reasoning="r", confidence=0.5)
        assert dec.action.source == "ai"

    def test_model_supplied_source_is_overridden(self):
        from ai_game_agent.ai import AIDecision

        # Even if the model claims a different origin, an AI decision is
        # AI-originated.
        payload = dict(_noop_action_dict(), source="user")
        dec = AIDecision.from_dict({"action": payload})
        assert dec.action.source == "ai"

    def test_roundtrip_preserves_ai_source(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        dec2 = AIDecision.from_dict(dec.to_dict())
        assert dec2.action.source == "ai"


class TestAIDecisionConfidenceConsistency:
    """The decision's confidence is the single source of truth.

    An ``action.confidence`` that contradicts the decision's confidence is
    ambiguous model output and must be rejected (fail loud) rather than
    silently normalized. A consistent (or absent) action confidence is
    accepted.
    """

    def test_conflicting_action_confidence_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        action = dict(_noop_action_dict(), confidence=0.3)
        with pytest.raises(AIError):
            AIDecision.from_dict({"action": action, "confidence": 0.9})

    def test_matching_action_confidence_accepted(self):
        from ai_game_agent.ai import AIDecision

        action = dict(_noop_action_dict(), confidence=0.7)
        dec = AIDecision.from_dict({"action": action, "confidence": 0.7})
        assert dec.confidence == 0.7

    def test_action_confidence_absent_accepted(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict(), "confidence": 0.4})
        assert dec.confidence == 0.4

    def test_direct_construction_conflicting_rejected(self):
        from ai_game_agent.actions import Action
        from ai_game_agent.ai import AIDecision, AIError

        action = Action.from_dict(dict(_noop_action_dict(), confidence=0.1))
        with pytest.raises(AIError):
            AIDecision(action=action, confidence=0.8)


class TestAIDecisionConfidenceType:
    """``AIDecision.confidence`` is always a ``float``, on every path.

    ``from_dict`` normalizes via ``float(confidence)`` but direct
    construction stored the value as-given, so an ``int`` confidence
    (legal in Python: ``isinstance(1, (int, float))``) produced
    ``dec.confidence == 1`` (int) from one path and ``1.0`` (float) from
    the other, and ``to_dict()`` serialized the same logical decision as
    ``1`` vs ``1.0``. The contract (schema.py: "Model-reported confidence
    in [0.0, 1.0]") and the field annotation (``float``) say float; both
    paths must agree.
    """

    def test_direct_construction_int_confidence_is_float(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision(action=_click_action(), confidence=1)
        assert dec.confidence == 1.0
        assert type(dec.confidence) is float

    def test_direct_construction_int_zero_confidence_is_float(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision(action=_click_action(), confidence=0)
        assert dec.confidence == 0.0
        assert type(dec.confidence) is float

    def test_both_construction_paths_serialize_identically(self):
        """Same logical decision, same JSON bytes, whichever path built it."""

        from ai_game_agent.actions import Action
        from ai_game_agent.ai import AIDecision

        payload = dict(_noop_action_dict(), confidence=1)
        direct = AIDecision(action=Action.from_dict(payload), confidence=1)
        via_dict = AIDecision.from_dict({"action": payload, "confidence": 1})
        assert json.dumps(direct.to_dict(), sort_keys=True) == json.dumps(
            via_dict.to_dict(), sort_keys=True
        )
        assert direct.to_dict()["confidence"] == via_dict.to_dict()["confidence"]
        assert type(direct.to_dict()["confidence"]) is float


def _click_action_dict() -> dict:
    """A valid mouse_click Action dict (Phase-3 canonical vocabulary)."""
    return {
        "kind": "mouse_click",
        "x": 100,
        "y": 200,
        "button": "left",
        "clicks": 1,
        "schema_version": 1,
    }


def _noop_action_dict() -> dict:
    """A minimal valid in-place mouse_click (no required fields)."""
    return {"kind": "mouse_click", "schema_version": 1}


class TestAIDecision:
    def test_from_dict_valid_click(self):
        from ai_game_agent.ai import AIDecision

        d = {"action": _click_action_dict(), "reasoning": "click target",
             "confidence": 0.92}
        dec = AIDecision.from_dict(d)
        assert dec.kind == "mouse_click"
        assert dec.confidence == 0.92
        assert dec.reasoning == "click target"
        assert dec.action.kind.value == "mouse_click"

    def test_from_dict_valid_noop(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        assert dec.kind == "mouse_click"
        assert dec.confidence == 1.0  # default
        assert dec.reasoning == ""

    def test_from_dict_missing_action_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict({"reasoning": "no action"})

    def test_from_dict_non_dict_action_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict({"action": "click"})

    def test_from_dict_invalid_action_kind_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict({"action": {"kind": "teleport",
                                              "schema_version": 1}})

    def test_from_dict_non_dict_top_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict([1, 2, 3])

    def test_from_dict_bad_confidence_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": 1.5}
            )

    def test_from_dict_confidence_string_raises(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": "high"}
            )

    def test_from_dict_confidence_bool_rejected(self):
        from ai_game_agent.ai import AIDecision, AIError

        with pytest.raises(AIError):
            AIDecision.from_dict(
                {"action": _noop_action_dict(), "confidence": True}
            )

    def test_to_dict_roundtrip(self):
        from ai_game_agent.ai import AIDecision

        d = {"action": _click_action_dict(), "reasoning": "r",
             "confidence": 0.5}
        dec = AIDecision.from_dict(d)
        out = dec.to_dict()
        dec2 = AIDecision.from_dict(out)
        assert dec2.action == dec.action
        assert dec2.reasoning == "r"
        assert dec2.confidence == 0.5

    def test_to_dict_shape(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        out = dec.to_dict()
        assert out["schema_version"] == 1
        assert "action" in out
        assert "kind" in out["action"]
        assert out["confidence"] == 1.0

    def test_frozen(self):
        from ai_game_agent.ai import AIDecision

        dec = AIDecision.from_dict({"action": _noop_action_dict()})
        with pytest.raises(AttributeError):
            dec.reasoning = "mutated"  # type: ignore[misc]
