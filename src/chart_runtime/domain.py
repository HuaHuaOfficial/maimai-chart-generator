"""Immutable identifiers and boundary data, with no chart legality logic."""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class CudaPayload(Protocol):
    """Owned device buffers; implementations validate their actual CUDA device.

    Python-side metadata is not evidence that tensors actually reside on CUDA.
    Concrete backends must check storage, dtype, schema and lease before use.
    """

    @property
    def device(self) -> str: ...

    @property
    def schema_id(self) -> str: ...


@dataclass(frozen=True)
class Definition:
    """Content-addressed semantics shared by model adapter and Harness."""

    chart_schema: str
    rules_digest: str
    features_digest: str
    calibration_digest: str
    path_tables_digest: str


@dataclass(frozen=True)
class ChartRef:
    request_id: str
    revision_id: str
    content_digest: str


@dataclass(frozen=True)
class Chart:
    ref: ChartRef
    definition: Definition
    payload: CudaPayload


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    version_id: int
    difficulty_slot: int
    ds_tenths: int
    seed: int
    definition: Definition
    # Controls include the tempo map, audio end, immutable user requirements,
    # audio features, and Star policy. A backend must not relax them implicitly.
    conditions: CudaPayload


@dataclass(frozen=True)
class Budget:
    max_rounds: int
    max_candidates: int
    workspace_bytes: int


class Scope(str, Enum):
    CANDIDATE = "candidate"
    FULL_CHART = "full_chart"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Proposal:
    chart: Chart
    base: ChartRef | None
    feedback_id: str
    complete: bool = True
    interruption: CudaPayload | None = None


@dataclass(frozen=True)
class Evaluation:
    chart: ChartRef
    definition: Definition
    scope: Scope
    verdict: Verdict
    coverage_complete: bool
    # Full device-side witnesses, not a truncated UI list or a count of errors.
    witnesses: CudaPayload
    receipt_id: str


@dataclass(frozen=True)
class Observation:
    proposal: Proposal
    evaluation: Evaluation


@dataclass(frozen=True)
class Feedback:
    feedback_id: str
    request_id: str
    definition: Definition
    # Harness can choose a rejected draft as the next working base without
    # declaring it accepted. This allows progress across equal error counts.
    base: ChartRef | None
    # Masks, affected scope and feasibility evidence are compiled by Harness.
    # They are tied to this base, not copied blindly between different drafts.
    constraints: CudaPayload
    edit_scope: CudaPayload
    # Explicit ledger: every observation considered by this feedback revision.
    observed_receipts: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackStop:
    request_id: str
    reason: str
    verdict: Verdict


@dataclass(frozen=True)
class PublishPermit:
    chart: ChartRef
    definition: Definition
    receipt_id: str


@dataclass(frozen=True)
class SessionResult:
    state: str
    observations: tuple[Observation, ...]
    chart: Chart | None = None
    permit: PublishPermit | None = None
    reason: str = ""
