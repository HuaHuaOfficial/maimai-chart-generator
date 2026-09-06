from typing import Protocol

from ..domain import Budget, Feedback, GenerationRequest, Proposal


class Generator(Protocol):
    def propose(
        self, request: GenerationRequest, feedback: Feedback, budget: Budget
    ) -> tuple[Proposal, ...]:
        """Create complete immutable drafts under the current Harness feedback.

        The internal implementation may plan, fill spans, resample or abstain
        where the edit scope allows. It cannot judge or publish its own result.
        Stable user requirements are never silently relaxed to obtain success.
        """
        ...
