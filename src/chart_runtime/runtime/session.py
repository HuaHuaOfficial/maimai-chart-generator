"""Rule-free coordinator between the two authority interfaces.

The production CUDA backends are not implemented by this reference loop.
It is exercised with protocol doubles to check revision and feedback handling.
"""
from ..domain import (
    Budget, FeedbackStop, GenerationRequest, Observation, Scope, SessionResult, Verdict,
)
from ..generator.port import Generator
from ..harness.port import Harness


class ContractError(RuntimeError):
    pass


def run_session(
    request: GenerationRequest, generator: Generator, harness: Harness, budget: Budget
) -> SessionResult:
    if min(budget.max_rounds, budget.max_candidates, budget.workspace_bytes) <= 0:
        raise ValueError("All session budgets must be positive")
    history: tuple[Observation, ...] = ()
    for _ in range(budget.max_rounds):
        feedback = harness.feedback(request, history, budget)
        if isinstance(feedback, FeedbackStop):
            if feedback.request_id != request.request_id:
                raise ContractError("Feedback stop belongs to another request")
            if feedback.verdict not in (Verdict.INFEASIBLE, Verdict.UNSUPPORTED):
                raise ContractError("A feedback stop needs an explicit terminal verdict")
            return SessionResult(feedback.verdict.value, history, reason=feedback.reason)
        if feedback.request_id != request.request_id or feedback.definition != request.definition:
            raise ContractError("Feedback definition/request mismatch")
        if feedback.observed_receipts != tuple(x.evaluation.receipt_id for x in history):
            raise ContractError("Feedback dropped or reordered observed evidence")
        known = {x.proposal.chart.ref for x in history}
        if feedback.base is not None and feedback.base not in known:
            raise ContractError("Feedback references a draft absent from session history")
        proposals = generator.propose(request, feedback, budget)
        if not proposals:
            return SessionResult("search_exhausted", history, reason="Generator returned no proposals")
        if len(proposals) > budget.max_candidates:
            raise ContractError("Generator exceeded the candidate budget")
        refs = tuple(x.chart.ref for x in proposals)
        if len(set(refs)) != len(refs) or any(ref in known for ref in refs):
            raise ContractError("Each proposal needs a fresh immutable revision")
        for proposal in proposals:
            if proposal.chart.ref.request_id != request.request_id:
                raise ContractError("Proposal belongs to another request")
            if proposal.chart.definition != request.definition:
                raise ContractError("Proposal changed rule/feature/calibration definitions")
            if proposal.base != feedback.base or proposal.feedback_id != feedback.feedback_id:
                raise ContractError("Generator used stale feedback or the wrong base")
        evaluations = harness.evaluate_batch(request, proposals, budget)
        if len(evaluations) != len(proposals):
            raise ContractError("Harness must return exactly one result per proposal")
        receipts = {x.evaluation.receipt_id for x in history}
        for proposal, evaluation in zip(proposals, evaluations):
            if evaluation.chart != proposal.chart.ref or evaluation.definition != request.definition:
                raise ContractError("Harness evaluated another draft or semantic definition")
            if evaluation.receipt_id in receipts:
                raise ContractError("Harness reused a receipt identifier")
            receipts.add(evaluation.receipt_id)
        # Keep ALL results, including equal counts, new failure categories,
        # and locally accepted proposals lacking whole-chart coverage.
        history += tuple(Observation(p, e) for p, e in zip(proposals, evaluations))
        for proposal, evaluation in zip(proposals, evaluations):
            if (proposal.complete and evaluation.verdict is Verdict.ACCEPT and
                    evaluation.scope is Scope.FULL_CHART and evaluation.coverage_complete):
                permit = harness.permit(evaluation)
                if (permit.chart != proposal.chart.ref or permit.definition != request.definition or
                        permit.receipt_id != evaluation.receipt_id):
                    raise ContractError("Publish permit does not bind to this full-chart receipt")
                return SessionResult("accepted", history, proposal.chart, permit)
    return SessionResult("budget_exhausted", history, reason="No full-chart acceptance within budget")
