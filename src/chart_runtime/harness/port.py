from typing import Protocol

from ..domain import (
    Budget, Evaluation, Feedback, FeedbackStop, GenerationRequest, Observation,
    Proposal, PublishPermit,
)


class Harness(Protocol):
    def feedback(
        self, request: GenerationRequest, history: tuple[Observation, ...], budget: Budget
    ) -> Feedback | FeedbackStop:
        """Compile joint constraints using the entire observed failure history.

        Re-evaluate dependent witnesses against the chosen base. An old witness
        is evidence to reconsider, not a permanent prohibition on all drafts.
        Diagnose incompatible constraints and expand scope when authorized;
        never infer infeasibility solely from a failed finite random search.
        """
        ...

    def evaluate_batch(
        self, request: GenerationRequest, proposals: tuple[Proposal, ...], budget: Budget
    ) -> tuple[Evaluation, ...]:
        """Run common CUDA rules/features for all proposals in bounded tiles.

        Candidate and full-chart scopes share semantics and precision. Only
        FULL_CHART + ACCEPT + complete coverage can produce a publish permit.
        Host code may pack data and report results, never judge events serially.
        """
        ...

    def permit(self, evaluation: Evaluation) -> PublishPermit:
        """Bind a full-chart acceptance receipt to the exact immutable draft."""
        ...
