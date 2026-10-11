"""Action queue: validated, ordered, de-duplicated action requests.

Phase 3, step 4. The queue accepts :class:`ActionRequest` objects (each
wrapping a validated :class:`~ai_game_agent.actions.action.Action` plus a
request timestamp), refuses to queue a second request of the same kind
while one is already pending, and hands requests to the backend FIFO via
:meth:`ActionQueue.pop_next`.

Validation is performed at construction time of the underlying :class:`Action`
(fail fast); the queue adds the ordering, de-duplication, and lifecycle
semantics on top.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .action import Action
from .base import ActionQueueError


@dataclass(frozen=True)
class ActionRequest:
    """One action request awaiting execution.

    ``action`` is the validated, immutable payload (already checked against
    the per-kind contract by :class:`Action.__post_init__`); ``requested_at``
    is the logical time the agent decided to enqueue it (used by the log and
    the executor for ordering/provenance).
    """

    action: Action
    requested_at: datetime


class ActionQueue:
    """FIFO queue of :class:`ActionRequest` bound to a backend.

    The queue rejects a duplicate pending kind (the earlier request wins) and
    returns requests oldest-first via :meth:`pop_next`. ``clear`` is idempotent
    and returns the number of dropped requests.

    The backend is retained so the queue can later gate ``pop_next`` on
    backend openness (step 5, executor) without changing the queue's contract.
    """

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._pending: deque[ActionRequest] = deque()

    # -- state --------------------------------------------------------

    @property
    def backend(self) -> object:
        """The backend this queue hands executed requests to."""
        return self._backend

    def pending_count(self) -> int:
        """Number of requests currently waiting to execute."""
        return len(self._pending)

    def pending_kinds(self) -> tuple[str, ...]:
        """Kinds of pending requests, FIFO order (as strings)."""
        return tuple(req.action.kind.value for req in self._pending)

    # -- mutation -----------------------------------------------------

    def add(self, request: ActionRequest) -> None:
        """Enqueue ``request``.

        Raises:
            ActionQueueError: if a request with the same kind is already
                pending (the earlier one wins).
        """
        kind = request.action.kind.value
        if kind in self.pending_kinds():
            raise ActionQueueError(
                f"duplicate pending kind {kind!r}; "
                "clear the queue before adding another of the same kind",
            )
        self._pending.append(request)

    def pop_next(self) -> ActionRequest | None:
        """Return and remove the oldest pending request, or ``None`` if empty."""
        if not self._pending:
            return None
        return self._pending.popleft()

    def clear(self) -> int:
        """Drop all pending requests; return how many were removed.

        Idempotent: a second call returns ``0`` and does not raise.
        """
        count = len(self._pending)
        self._pending.clear()
        return count

    def cancel_all(self) -> int:
        """Cancel every pending request (plan §5.3 naming).

        Alias of :meth:`clear`: the emergency stop and other priority paths
        call this so they read as "cancel queued work." Returns the number of
        cancelled requests; idempotent.
        """
        return self.clear()
