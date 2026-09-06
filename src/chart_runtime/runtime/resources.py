"""CUDA workspace admission and bounded tiling.

This module owns only execution resources.  It deliberately contains no chart
or musical legality logic and never changes process-wide torch settings.  A
workspace manager is scoped to one CUDA device and admits shared leases by
accounting requested workspace bytes against a conservative, live budget.

The default live budget is the smaller of the current free memory after an
operational reserve and a bounded working cap.  This is intentionally more
conservative than treating ``mem_get_info``'s free value as an allocation
promise: another framework allocator may consume memory between the probe and
the actual kernel launch, so callers still need normal CUDA OOM handling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import threading
import time
from typing import Any, Callable, Iterator, Optional

try:  # Importing the runtime should still be possible in a CPU-only test env.
    import torch  # type: ignore
except Exception:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None  # type: ignore[assignment]


MiB = 1024 * 1024
DEFAULT_MAX_WORKING_BYTES = 256 * MiB
DEFAULT_MAX_WORKING_FRACTION = 0.25
DEFAULT_OPERATIONAL_RESERVE_BYTES = 64 * MiB
DEFAULT_TILE_CAP = 4096


class ResourceBudgetError(RuntimeError):
    """A requested CUDA workspace cannot be admitted safely."""


class UnsupportedDeviceError(ResourceBudgetError, ValueError):
    """The resource manager was asked to operate on a non-CUDA device."""


class CudaUnavailableError(ResourceBudgetError):
    """CUDA or the live CUDA memory query is unavailable."""


class LeaseCancelledError(ResourceBudgetError):
    """A waiting lease observed its cancellation event."""


class LeaseTimeoutError(ResourceBudgetError):
    """A waiting lease exceeded its optional admission timeout."""


# Friendly aliases for callers that use the more explicit spelling.
WorkspaceCancelledError = LeaseCancelledError
WorkspaceTimeoutError = LeaseTimeoutError


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """The live accounting view used for one admission or tile decision."""

    device: str
    free_bytes: int
    total_bytes: int
    operational_reserve_bytes: int
    max_working_bytes: int
    effective_budget_bytes: int
    leased_bytes: int
    available_bytes: int

    @property
    def live_budget_bytes(self) -> int:
        """The live budget before subtracting this manager's active leases."""

        return self.effective_budget_bytes


@dataclass(frozen=True)
class WorkspaceTelemetry:
    """Optional observations; these are not a claim of total GPU capacity.

    ``peak_torch_*`` are samples of PyTorch's device counters while leases are
    active.  They are intentionally named as PyTorch counters rather than
    workspace usage because the manager does not own allocations made by the
    framework.  ``lease_max_bytes`` is the largest admitted lease request seen
    by this manager.
    """

    device: str
    peak_torch_allocated_bytes: Optional[int]
    peak_torch_reserved_bytes: Optional[int]
    lease_max_bytes: int
    active_lease_bytes: int
    sample_count: int

    @property
    def peak_allocated_bytes(self) -> Optional[int]:
        return self.peak_torch_allocated_bytes

    @property
    def peak_reserved_bytes(self) -> Optional[int]:
        return self.peak_torch_reserved_bytes

    @property
    def lease_max(self) -> int:
        return self.lease_max_bytes


@dataclass(frozen=True)
class TilePlan:
    """A bounded row chunk and its conservative workspace accounting."""

    rows: int
    columns: int
    bytes_per_cell: int
    intermediates: int
    bytes_per_row: int
    workspace_bytes: int
    available_bytes: int
    cap: int


class WorkspaceLease:
    """Metadata for an admitted lease.

    The object is yielded by :meth:`WorkspaceManager.lease`.  It does not own
    a tensor or change a torch allocator; it only marks a byte reservation
    until the context exits.
    """

    __slots__ = ("_manager", "_requested_bytes", "_stream", "_released")

    def __init__(self, manager: "WorkspaceManager", requested_bytes: int, stream: Any) -> None:
        self._manager = manager
        self._requested_bytes = requested_bytes
        self._stream = stream
        self._released = False

    @property
    def device(self) -> str:
        return self._manager.device

    @property
    def requested_bytes(self) -> int:
        return self._requested_bytes

    @property
    def released(self) -> bool:
        return self._released

    @property
    def stream(self) -> Any:
        return self._stream


def _require_integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _call_with_optional_device(function: Callable[..., Any], device: str) -> Any:
    """Call an injected one-argument or zero-argument test hook."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(device)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL
           for parameter in signature.parameters.values()) or positional:
        return function(device)
    return function()


def _call_sync(function: Callable[..., Any], device: str, stream: Any) -> Any:
    """Call a sync hook with ``(device, stream)``, ``(device)`` or no args."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(device, stream)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL
           for parameter in signature.parameters.values()) or len(positional) >= 2:
        return function(device, stream)
    if len(positional) == 1:
        name = positional[0].name.lower()
        return function(stream if "stream" in name else device)
    return function()


def _normalize_device(device: Any) -> str:
    """Normalize an explicit CUDA device without silently selecting device 0."""

    if hasattr(device, "type"):
        device_type = str(device.type).lower()
        index = getattr(device, "index", None)
    elif isinstance(device, str):
        value = device.strip().lower()
        if ":" in value:
            device_type, raw_index = value.split(":", 1)
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise UnsupportedDeviceError(f"Invalid CUDA device {device!r}") from exc
        else:
            device_type, index = value, None
    else:
        raise UnsupportedDeviceError(
            f"CUDA workspace requires a CUDA device, got {device!r}; CPU fallback is forbidden"
        )

    if device_type != "cuda":
        raise UnsupportedDeviceError(
            f"CUDA workspace requires a CUDA device, got {device!r}; CPU fallback is forbidden"
        )
    if index is None:
        if torch is None or not hasattr(torch, "cuda") or not hasattr(torch.cuda, "current_device"):
            raise CudaUnavailableError(
                "An unindexed CUDA device needs torch.cuda.current_device(); no CPU fallback is allowed"
            )
        try:
            index = int(torch.cuda.current_device())
        except Exception as exc:
            raise CudaUnavailableError("Could not resolve the current CUDA device") from exc
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise UnsupportedDeviceError(f"Invalid CUDA device index {index!r}")
    return f"cuda:{index}"


class WorkspaceManager:
    """Thread-safe shared CUDA workspace budget for one device.

    The manager uses an :class:`~threading.RLock` plus a condition variable for
    admission.  Multiple leases may coexist when their byte reservations fit;
    a later lease waits for an active lease to release instead of changing
    torch's global allocator configuration.  Synchronization happens while the
    per-device lock is held, before the reservation is returned to the pool.
    """

    def __init__(
        self,
        device: Any = "cuda:0",
        *,
        operational_reserve_bytes: int = DEFAULT_OPERATIONAL_RESERVE_BYTES,
        max_working_bytes: Optional[int] = None,
        mem_info_fn: Optional[Callable[..., Any]] = None,
        availability_fn: Optional[Callable[..., Any]] = None,
        synchronize_fn: Optional[Callable[..., Any]] = None,
        telemetry: bool = False,
    ) -> None:
        self._device = _normalize_device(device)
        self._operational_reserve_bytes = _require_integer(
            operational_reserve_bytes, "operational_reserve_bytes"
        )
        if max_working_bytes is not None:
            max_working_bytes = _require_integer(max_working_bytes, "max_working_bytes", positive=True)
        if mem_info_fn is not None and availability_fn is not None:
            raise ValueError("pass only one of mem_info_fn and availability_fn")
        self._configured_max_working_bytes = max_working_bytes
        self._mem_info_fn = mem_info_fn or availability_fn or self._default_mem_info
        self._synchronize_fn = synchronize_fn
        self._telemetry_enabled = bool(telemetry)

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._leased_bytes = 0
        self._lease_max_bytes = 0
        self._peak_torch_allocated: Optional[int] = None
        self._peak_torch_reserved: Optional[int] = None
        self._telemetry_samples = 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def torch_device(self) -> Any:
        if torch is None or not hasattr(torch, "device"):
            return self._device
        return torch.device(self._device)

    @property
    def operational_reserve_bytes(self) -> int:
        return self._operational_reserve_bytes

    @property
    def configured_max_working_bytes(self) -> Optional[int]:
        return self._configured_max_working_bytes

    @property
    def active_lease_bytes(self) -> int:
        with self._lock:
            return self._leased_bytes

    @property
    def lease_max_bytes(self) -> int:
        with self._lock:
            return self._lease_max_bytes

    @property
    def lease_max(self) -> int:
        return self.lease_max_bytes

    def _default_mem_info(self, device: str) -> tuple[int, int]:
        if torch is None or not hasattr(torch, "cuda") or not hasattr(torch.cuda, "mem_get_info"):
            raise CudaUnavailableError("torch CUDA memory information is unavailable; CPU fallback is forbidden")
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception as exc:
            raise CudaUnavailableError(
                f"Could not query live CUDA memory for {device}; CPU fallback is forbidden"
            ) from exc
        return int(free_bytes), int(total_bytes)

    def _default_synchronize(self, device: str, stream: Any) -> None:
        if stream is not None and hasattr(stream, "synchronize"):
            stream.synchronize()
            return
        if torch is None or not hasattr(torch, "cuda"):
            raise CudaUnavailableError("CUDA synchronization is unavailable; CPU fallback is forbidden")
        # Prefer the stream currently used by the caller.  The explicit stream
        # parameter is available for callers launching on a non-current stream.
        if hasattr(torch.cuda, "current_stream"):
            try:
                current_stream = torch.cuda.current_stream(device)
                if hasattr(current_stream, "synchronize"):
                    current_stream.synchronize()
                    return
            except Exception as exc:
                raise CudaUnavailableError(f"Could not synchronize CUDA stream on {device}") from exc
        if hasattr(torch.cuda, "synchronize"):
            try:
                torch.cuda.synchronize(device)
                return
            except Exception as exc:
                raise CudaUnavailableError(f"Could not synchronize CUDA device {device}") from exc
        raise CudaUnavailableError("CUDA synchronization is unavailable; CPU fallback is forbidden")

    def _live_memory_unlocked(self) -> tuple[int, int]:
        try:
            raw = _call_with_optional_device(self._mem_info_fn, self._device)
            free_bytes, total_bytes = raw
            free_bytes = _require_integer(int(free_bytes), "free_bytes")
            total_bytes = _require_integer(int(total_bytes), "total_bytes")
        except ResourceBudgetError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(f"Could not query live CUDA memory for {self._device}") from exc
        if total_bytes < free_bytes:
            # A malformed/injected query must not enlarge the budget.
            free_bytes = total_bytes
        return free_bytes, total_bytes

    def _effective_cap_unlocked(self, total_bytes: int) -> int:
        if self._configured_max_working_bytes is not None:
            return self._configured_max_working_bytes
        # A bounded fraction keeps tiny test/dev devices usable while keeping
        # the default on large cards near 256 MiB.
        fractional_cap = int(total_bytes * DEFAULT_MAX_WORKING_FRACTION)
        return min(DEFAULT_MAX_WORKING_BYTES, max(1, fractional_cap))

    def _snapshot_unlocked(self) -> WorkspaceSnapshot:
        free_bytes, total_bytes = self._live_memory_unlocked()
        max_working_bytes = self._effective_cap_unlocked(total_bytes)
        after_reserve = max(0, free_bytes - self._operational_reserve_bytes)
        effective_budget = min(after_reserve, max_working_bytes)
        return WorkspaceSnapshot(
            device=self._device,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            operational_reserve_bytes=self._operational_reserve_bytes,
            max_working_bytes=max_working_bytes,
            effective_budget_bytes=effective_budget,
            leased_bytes=self._leased_bytes,
            available_bytes=max(0, effective_budget - self._leased_bytes),
        )

    def snapshot(self) -> WorkspaceSnapshot:
        """Return a fresh live budget snapshot for this device."""

        with self._lock:
            return self._snapshot_unlocked()

    def _sample_telemetry_unlocked(self) -> None:
        if not self._telemetry_enabled:
            return
        self._telemetry_samples += 1
        if torch is None or not hasattr(torch, "cuda"):
            return
        try:
            allocated = int(torch.cuda.memory_allocated(self._device))
        except Exception:
            allocated = None
        try:
            reserved = int(torch.cuda.memory_reserved(self._device))
        except Exception:
            reserved = None
        if allocated is not None:
            self._peak_torch_allocated = (
                allocated
                if self._peak_torch_allocated is None
                else max(self._peak_torch_allocated, allocated)
            )
        if reserved is not None:
            self._peak_torch_reserved = (
                reserved
                if self._peak_torch_reserved is None
                else max(self._peak_torch_reserved, reserved)
            )

    def telemetry(self) -> WorkspaceTelemetry:
        """Return sampled PyTorch counters and actual admitted lease maximum."""

        with self._lock:
            self._sample_telemetry_unlocked()
            return WorkspaceTelemetry(
                device=self._device,
                peak_torch_allocated_bytes=self._peak_torch_allocated,
                peak_torch_reserved_bytes=self._peak_torch_reserved,
                lease_max_bytes=self._lease_max_bytes,
                active_lease_bytes=self._leased_bytes,
                sample_count=self._telemetry_samples,
            )

    def get_telemetry(self) -> WorkspaceTelemetry:
        return self.telemetry()

    def _is_cancelled(self, cancel_event: Any) -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def _acquire(
        self,
        requested_bytes: int,
        *,
        cancel_event: Any,
        timeout: Optional[float],
        stream: Any,
    ) -> WorkspaceLease:
        requested_bytes = _require_integer(requested_bytes, "requested_bytes", positive=True)
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a non-negative number or None")
            if timeout < 0:
                raise ValueError("timeout must be non-negative")
        if self._is_cancelled(cancel_event):
            raise LeaseCancelledError("CUDA workspace lease was cancelled before admission")

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while True:
                if self._is_cancelled(cancel_event):
                    raise LeaseCancelledError("CUDA workspace lease was cancelled while waiting")
                snapshot = self._snapshot_unlocked()
                if requested_bytes > snapshot.effective_budget_bytes:
                    raise ResourceBudgetError(
                        f"Requested workspace {requested_bytes} bytes cannot fit one lease on "
                        f"{self._device}: live budget is {snapshot.effective_budget_bytes} bytes "
                        f"(free={snapshot.free_bytes}, reserve={snapshot.operational_reserve_bytes}, "
                        f"max_working={snapshot.max_working_bytes})"
                    )
                if self._leased_bytes + requested_bytes <= snapshot.effective_budget_bytes:
                    self._leased_bytes += requested_bytes
                    self._lease_max_bytes = max(self._lease_max_bytes, requested_bytes)
                    self._sample_telemetry_unlocked()
                    return WorkspaceLease(self, requested_bytes, stream)

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LeaseTimeoutError(
                            f"Timed out waiting for {requested_bytes} bytes on {self._device}"
                        )
                    wait_for = min(remaining, 0.1)
                else:
                    wait_for = 0.1
                self._condition.wait(wait_for)

    def _release(self, lease: WorkspaceLease) -> None:
        with self._condition:
            if lease._released:
                return
            synchronization_error: Optional[BaseException] = None
            try:
                if self._synchronize_fn is None:
                    self._default_synchronize(self._device, lease.stream)
                else:
                    _call_sync(self._synchronize_fn, self._device, lease.stream)
            except BaseException as exc:  # Always return the accounting reservation.
                synchronization_error = exc
            finally:
                self._leased_bytes -= lease.requested_bytes
                if self._leased_bytes < 0:  # Defensive invariant guard for custom misuse.
                    self._leased_bytes = 0
                lease._released = True
                self._sample_telemetry_unlocked()
                self._condition.notify_all()
            if synchronization_error is not None:
                raise synchronization_error

    @contextmanager
    def lease(
        self,
        requested_bytes: int,
        *,
        cancel_event: Any = None,
        timeout: Optional[float] = None,
        stream: Any = None,
    ) -> Iterator[WorkspaceLease]:
        """Admit a shared byte lease and synchronize before returning it.

        ``cancel_event`` is checked while waiting for another lease to release;
        it cannot interrupt a CUDA kernel already launched by the caller.  Pass
        the launch stream when it is not the current stream so release waits on
        the correct stream before admitting another operation.
        """

        admitted = self._acquire(
            requested_bytes,
            cancel_event=cancel_event,
            timeout=timeout,
            stream=stream,
        )
        try:
            yield admitted
        finally:
            self._release(admitted)

    def tile_plan(
        self,
        columns: int,
        *,
        bytes_per_cell: int = 4,
        intermediates: int = 1,
        cap: int = DEFAULT_TILE_CAP,
    ) -> TilePlan:
        """Choose a bounded row chunk from the currently available budget.

        A row costs ``columns * bytes_per_cell * intermediates`` bytes.  Thus
        both matrix dimensions participate in the accounting: ``columns`` is
        the full second dimension and ``rows`` is the selected first dimension.
        ``cap`` limits rows independently of the memory budget.
        """

        columns = _require_integer(columns, "columns", positive=True)
        bytes_per_cell = _require_integer(bytes_per_cell, "bytes_per_cell", positive=True)
        intermediates = _require_integer(intermediates, "intermediates", positive=True)
        cap = _require_integer(cap, "cap", positive=True)
        bytes_per_row = columns * bytes_per_cell * intermediates
        with self._lock:
            snapshot = self._snapshot_unlocked()
            if bytes_per_row > snapshot.available_bytes:
                raise ResourceBudgetError(
                    f"Cannot fit a single row on {self._device}: row needs {bytes_per_row} bytes, "
                    f"available workspace is {snapshot.available_bytes} bytes after active leases"
                )
            rows = min(cap, snapshot.available_bytes // bytes_per_row)
            if rows < 1:  # Keep the single-row failure explicit if arithmetic changes later.
                raise ResourceBudgetError(
                    f"Cannot fit a single row on {self._device}: row needs {bytes_per_row} bytes"
                )
            return TilePlan(
                rows=rows,
                columns=columns,
                bytes_per_cell=bytes_per_cell,
                intermediates=intermediates,
                bytes_per_row=bytes_per_row,
                workspace_bytes=rows * bytes_per_row,
                available_bytes=snapshot.available_bytes,
                cap=cap,
            )

    def tile_rows(
        self,
        columns: int,
        bytes_per_cell: int = 4,
        intermediates: int = 1,
        cap: int = DEFAULT_TILE_CAP,
    ) -> int:
        """Return the bounded row count; fail before any allocation if row 1 cannot fit."""

        return self.tile_plan(
            columns,
            bytes_per_cell=bytes_per_cell,
            intermediates=intermediates,
            cap=cap,
        ).rows


# The longer names are useful at integration boundaries and preserve a single
# implementation/state type for callers that prefer CUDA-specific naming.
CudaWorkspaceManager = WorkspaceManager
CUDAWorkspaceManager = WorkspaceManager


_WORKSPACE_LOCK = threading.RLock()
_WORKSPACES: dict[str, WorkspaceManager] = {}


def get_workspace(
    device: Any = "cuda:0",
    *,
    operational_reserve_bytes: int = DEFAULT_OPERATIONAL_RESERVE_BYTES,
    max_working_bytes: Optional[int] = None,
    mem_info_fn: Optional[Callable[..., Any]] = None,
    availability_fn: Optional[Callable[..., Any]] = None,
    synchronize_fn: Optional[Callable[..., Any]] = None,
    telemetry: bool = False,
) -> WorkspaceManager:
    """Return the process-local singleton manager for one explicit CUDA device.

    Optional configuration is used only when the singleton is first created;
    later calls for the same device return that exact manager.  Use
    :func:`clear_workspaces` in isolated tests, or construct ``WorkspaceManager``
    directly when multiple independent budgets are needed in one process.
    """

    normalized = _normalize_device(device)
    with _WORKSPACE_LOCK:
        manager = _WORKSPACES.get(normalized)
        if manager is None:
            manager = WorkspaceManager(
                normalized,
                operational_reserve_bytes=operational_reserve_bytes,
                max_working_bytes=max_working_bytes,
                mem_info_fn=mem_info_fn,
                availability_fn=availability_fn,
                synchronize_fn=synchronize_fn,
                telemetry=telemetry,
            )
            _WORKSPACES[normalized] = manager
        return manager


def clear_workspaces(device: Any = None) -> None:
    """Clear singleton managers; intended for test/process lifecycle teardown."""

    with _WORKSPACE_LOCK:
        if device is None:
            _WORKSPACES.clear()
            return
        _WORKSPACES.pop(_normalize_device(device), None)


__all__ = [
    "CUDAWorkspaceManager",
    "CudaUnavailableError",
    "CudaWorkspaceManager",
    "DEFAULT_MAX_WORKING_BYTES",
    "DEFAULT_MAX_WORKING_FRACTION",
    "DEFAULT_OPERATIONAL_RESERVE_BYTES",
    "DEFAULT_TILE_CAP",
    "LeaseCancelledError",
    "LeaseTimeoutError",
    "ResourceBudgetError",
    "TilePlan",
    "UnsupportedDeviceError",
    "WorkspaceCancelledError",
    "WorkspaceLease",
    "WorkspaceManager",
    "WorkspaceSnapshot",
    "WorkspaceTelemetry",
    "WorkspaceTimeoutError",
    "clear_workspaces",
    "get_workspace",
]
