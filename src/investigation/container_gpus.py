"""Strict NVIDIA device selector parsing for Docker argument vectors."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import math
import logging
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Callable

from src.investigation.gpu_quarantine import GpuQuarantine, quarantine_marker, read_quarantine

_GPU_DEVICE_RE = re.compile(r"^(?:[0-9]+|GPU-[A-Fa-f0-9-]+|MIG-[A-Za-z0-9/.-]+)$")
_STABLE_GPU_DEVICE_RE = re.compile(
    r"^(?:GPU-[A-Fa-f0-9-]+|MIG-(?:[A-Fa-f0-9-]+|GPU-[A-Fa-f0-9-]+/[0-9]+/[0-9]+))$"
)
GPU_ALLOCATION_BUSY_CATEGORY = "gpu_allocation_busy"
GPU_ALLOCATION_INVALID_CATEGORY = "gpu_allocation_invalid"
GPU_ALLOCATION_QUARANTINED_CATEGORY = "gpu_allocation_quarantined"
GPU_ALLOCATION_FAILURE_CATEGORIES = frozenset({
    GPU_ALLOCATION_BUSY_CATEGORY,
    GPU_ALLOCATION_INVALID_CATEGORY,
    GPU_ALLOCATION_QUARANTINED_CATEGORY,
})
GPU_ALLOCATION_LEASE_ROOT = Path("/tmp") / f"agentic-scientist-gpu-leases-{os.getuid()}"
GPU_SELECTOR_ENV = "AGENT_CONTAINER_GPUS"
GPU_POOL_ENV = "AGENT_CONTAINER_GPU_POOL"


class GpuSelectorError(ValueError):
    """Raised when an explicit Docker GPU selector is malformed or insufficient."""


class GpuLeaseError(RuntimeError):
    """Raised when a GPU allocation lease cannot be established safely."""


class GpuLeaseTimeout(GpuLeaseError):
    """Raised when a GPU allocation remains leased until the caller's deadline."""


class GpuLeaseQuarantined(GpuLeaseError):
    """Raised when a GPU was quarantined after cleanup could not be verified."""

    def __init__(self, message: str, *, quarantines: tuple[GpuQuarantine, ...] = ()) -> None:
        super().__init__(message)
        self.quarantines = quarantines


class GpuAllocationLease:
    """An acquired advisory lease over stable GPU device identifiers."""

    def __init__(
        self, devices: tuple[str, ...], descriptors: list[int], *, lease_root: Path | None = None,
    ) -> None:
        self._devices = devices
        self._descriptors = descriptors
        self._lease_root = lease_root
        self._runtime_id: str | None = None

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    def prepare_runtime(self) -> None:
        """Persist ownership before any evaluator, broker, or Docker launcher can start."""
        from .gpu_lease_recovery import mark_runtime, validate_held_allocation
        from .gpu_runtime_journal import RuntimeJournalError, create_runtime
        if not self._descriptors or self._lease_root is None or self._runtime_id is not None:
            raise GpuLeaseError("GPU runtime ownership cannot be prepared in this lease state")
        try:
            validate_held_allocation(self._lease_root, self._devices, self._descriptors)
            self._runtime_id = create_runtime(self._lease_root, self._devices)
            mark_runtime(self._descriptors, self._runtime_id)
            validate_held_allocation(self._lease_root, self._devices, self._descriptors)
        except (RuntimeJournalError, OSError) as exc:
            raise GpuLeaseError("GPU runtime ownership could not be persisted") from exc

    def wrap_command(self, command: list[str]) -> list[str]:
        from .gpu_lease_recovery import owned_command, validate_held_allocation
        if self._runtime_id is None or self._lease_root is None or not self._descriptors:
            raise GpuLeaseError("GPU runtime ownership must precede launch")
        validate_held_allocation(self._lease_root, self._devices, self._descriptors)
        return owned_command(self._lease_root, self._runtime_id, command)

    def quarantine(self) -> None:
        """Keep every leased device unavailable until verified runtime or operator recovery."""
        if not self._descriptors:
            raise GpuLeaseError("cannot quarantine a released GPU allocation lease")
        try:
            marker = quarantine_marker()
            for descriptor in self._descriptors:
                if os.pread(descriptor, 1, 0):
                    continue
                written = os.pwrite(descriptor, marker, 0)
                if written != len(marker):
                    raise OSError("short GPU quarantine marker write")
                os.ftruncate(descriptor, len(marker))
                os.fsync(descriptor)
        except OSError as exc:
            raise GpuLeaseError("could not persist GPU allocation quarantine") from exc

    def release(self) -> None:
        recovery_error: BaseException | None = None
        if self._runtime_id is not None and self._descriptors:
            from .gpu_lease_recovery import recover_group
            from .gpu_owner_process import ProcessIdentityError, current_identity
            from .gpu_runtime_cleanup import RuntimeRecoveryError
            from .gpu_runtime_journal import RuntimeJournalError
            assert self._lease_root is not None
            try:
                recover_group(
                    self._lease_root, self._runtime_id, self._devices, self._descriptors,
                    deadline=time.monotonic() + 30, releasing_owner=current_identity(),
                )
            except (
                RuntimeRecoveryError, RuntimeJournalError, ProcessIdentityError, GpuLeaseError,
                OSError,
            ):
                recovery_error = GpuLeaseQuarantined("GPU runtime cleanup remains unverified")
            except BaseException as exc:
                recovery_error = exc
        descriptors, self._descriptors = self._descriptors, []
        first_error: OSError | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError as exc:
                first_error = first_error or exc
        if recovery_error is not None:
            raise recovery_error
        if first_error is not None:
            raise GpuLeaseError("could not release GPU allocation lease") from first_error

    def __enter__(self) -> GpuAllocationLease:
        return self

    def __exit__(self, _type: object, failure: object, _traceback: object) -> None:
        try:
            self.release()
        except GpuLeaseQuarantined as cleanup_failure:
            if isinstance(failure, Exception):
                raise cleanup_failure from failure
            if failure is None:
                raise
            logging.getLogger(__name__).warning("GPU cleanup unverified; quarantine retained")


def acquire_gpu_allocation_lease(
    raw_selector: str | None,
    gpu_count: int,
    *,
    lease_root: Path,
    deadline: float,
    poll_interval_seconds: float = 0.05,
    recovery_deadline: float | None = None,
) -> GpuAllocationLease:
    """Acquire an advisory lease for a selector's stable GPU identifiers."""
    if not math.isfinite(deadline):
        raise GpuLeaseError("GPU lease deadline must be finite")
    if recovery_deadline is not None and not math.isfinite(recovery_deadline):
        raise GpuLeaseError("GPU recovery deadline must be finite")
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise GpuLeaseError("GPU lease poll interval must be positive and finite")
    devices = _stable_gpu_devices(raw_selector, gpu_count)
    if deadline <= time.monotonic():
        raise _lease_timeout(devices)
    descriptors = _open_gpu_lock_files(lease_root, devices)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _lease_timeout(devices)
            acquired = _try_lock_all(descriptors)
            if acquired:
                from .gpu_lease_recovery import validate_held_allocation
                validate_held_allocation(lease_root, devices, descriptors)
                _recover_owned_devices(
                    lease_root, devices, descriptors,
                    deadline=min(recovery_deadline or deadline, time.monotonic() + 30),
                )
                _reject_quarantined_devices(devices, descriptors)
                return GpuAllocationLease(devices, descriptors, lease_root=lease_root)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _lease_timeout(devices)
            time.sleep(min(poll_interval_seconds, remaining))
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _lease_timeout(devices: tuple[str, ...]) -> GpuLeaseTimeout:
    return GpuLeaseTimeout(
        "GPU allocation remained busy until the deadline: " + ", ".join(devices)
    )


def _recover_owned_devices(
    root: Path, devices: tuple[str, ...], descriptors: list[int], *, deadline: float,
) -> None:
    from .gpu_lease_recovery import recover_group
    from .gpu_runtime_cleanup import RuntimeRecoveryError
    from .gpu_runtime_journal import RuntimeJournalError
    attempted: set[str] = set()
    for device, descriptor in zip(devices, descriptors, strict=True):
        marker = read_quarantine(descriptor, device)
        if marker is None or marker.recovery_id is None or marker.recovery_id in attempted:
            continue
        attempted.add(marker.recovery_id)
        try:
            recover_group(root, marker.recovery_id, devices, descriptors, deadline=deadline)
            logging.getLogger(__name__).info("Recovered abandoned GPU runtime %s", marker.recovery_id)
        except GpuLeaseTimeout:
            raise
        except (RuntimeRecoveryError, RuntimeJournalError, GpuLeaseError, OSError) as exc:
            diagnostic = (
                str(exc) if isinstance(exc, (RuntimeRecoveryError, RuntimeJournalError))
                else type(exc).__name__
            )
            logging.getLogger(__name__).warning(
                "GPU runtime %s remains quarantined: %s", marker.recovery_id, diagnostic,
            )


def _reject_quarantined_devices(devices: tuple[str, ...], descriptors: list[int]) -> None:
    try:
        quarantines = tuple(
            record
            for device, descriptor in zip(devices, descriptors, strict=True)
            if (record := read_quarantine(descriptor, device)) is not None
        )
    except OSError as exc:
        raise GpuLeaseError("could not inspect GPU allocation quarantine") from exc
    if quarantines:
        raise GpuLeaseQuarantined(
            "GPU allocation remains quarantined after recovery checks: "
            + ", ".join(record.device for record in quarantines),
            quarantines=quarantines,
        )


def selector_devices(raw_selector: str | None) -> tuple[str, ...]:
    """The devices a ``device=<ids>`` selector names; raises ``GpuSelectorError`` when malformed."""
    return _selector_devices(raw_selector)


def _selector_devices(raw_selector: str | None) -> tuple[str, ...]:
    selector = _unquote_selector(raw_selector)
    if not selector.startswith("device="):
        raise GpuSelectorError("GPU selector must use device=<ids>")
    devices = tuple(selector.removeprefix("device=").split(","))
    if not devices or any(not device or device.strip() != device for device in devices):
        raise GpuSelectorError("GPU selector contains an empty or padded device")
    if len(devices) != len(set(devices)):
        raise GpuSelectorError("GPU selector contains duplicate devices")
    if any(_GPU_DEVICE_RE.fullmatch(device) is None for device in devices):
        raise GpuSelectorError("GPU selector contains an invalid device")
    return devices


def _stable_gpu_devices(raw_selector: str | None, gpu_count: int) -> tuple[str, ...]:
    if gpu_count <= 0:
        raise GpuLeaseError("GPU lease count must be positive")
    try:
        devices = _selector_devices(raw_selector)
    except GpuSelectorError as exc:
        raise GpuLeaseError(f"invalid GPU lease selector: {exc}") from exc
    if len(devices) < gpu_count:
        raise GpuLeaseError(
            f"GPU lease requires {gpu_count} GPUs but selector exposes {len(devices)}"
        )
    selected = devices[:gpu_count]
    if any(_STABLE_GPU_DEVICE_RE.fullmatch(device) is None for device in selected):
        raise GpuLeaseError("GPU allocation leases require stable GPU UUID device selectors")
    canonical = [device.casefold() for device in selected]
    if len(canonical) != len(set(canonical)):
        raise GpuLeaseError("GPU lease selector contains duplicate stable device identifiers")
    return tuple(sorted(selected, key=str.casefold))


def _open_gpu_lock_files(lease_root: Path, devices: tuple[str, ...]) -> list[int]:
    try:
        lease_root.mkdir(mode=0o700, exist_ok=True)
        root_descriptor = os.open(
            lease_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise GpuLeaseError("GPU lease root is unavailable or unsafe") from exc
    descriptors: list[int] = []
    try:
        root_metadata = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.getuid()
            or root_metadata.st_mode & 0o077
        ):
            raise GpuLeaseError("GPU lease root must be an owner-controlled directory")
        for device in devices:
            filename = hashlib.sha256(device.casefold().encode("ascii")).hexdigest() + ".lock"
            descriptor = os.open(
                filename,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=root_descriptor,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
            ):
                os.close(descriptor)
                raise GpuLeaseError("GPU lease file must be a private owner-controlled file")
            descriptors.append(descriptor)
    except OSError as exc:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise GpuLeaseError("GPU lease lock file is unavailable or unsafe") from exc
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(root_descriptor)
    return descriptors


def _try_lock_all(descriptors: list[int]) -> bool:
    acquired: list[int] = []
    try:
        for descriptor in descriptors:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise GpuLeaseError("GPU lease lock operation failed") from exc
                return False
            acquired.append(descriptor)
        return True
    finally:
        if len(acquired) != len(descriptors):
            for descriptor in reversed(acquired):
                fcntl.flock(descriptor, fcntl.LOCK_UN)


def _unquote_selector(raw_selector: str | None) -> str:
    selector = (raw_selector or "").strip()
    starts_quoted = selector.startswith('"')
    ends_quoted = selector.endswith('"')
    if starts_quoted != ends_quoted:
        raise GpuSelectorError("GPU selector has unbalanced quotes")
    if starts_quoted:
        selector = selector[1:-1]
    if '"' in selector:
        raise GpuSelectorError("GPU selector contains an unexpected quote")
    return selector


def validate_gpu_allocation(
    raw_selector: str | None, gpu_count: int, *, label: str | None = None
) -> tuple[str, ...]:
    """Check that *raw_selector* exposes exactly *gpu_count* stable GPU UUIDs.

    A zero count requires an unset or empty selector so CPU-only runs never hand
    GPUs to agent containers; a positive count requires exactly that many unique
    stable UUID devices (the identifiers GPU allocation leases accept). *label*
    names the requirement in diagnostics (default ``container.gpu_count=<n>``).
    """
    if type(gpu_count) is not int or gpu_count < 0:
        raise GpuSelectorError("GPU count must be a non-negative integer")
    label = label or f"container.gpu_count={gpu_count}"
    if gpu_count == 0:
        if raw_selector is not None and raw_selector.strip():
            raise GpuSelectorError(
                f"{GPU_SELECTOR_ENV} is set but {label}; unset it or declare the GPU count"
            )
        return ()
    if raw_selector is None or not raw_selector.strip():
        raise GpuSelectorError(
            f"{label} requires {GPU_SELECTOR_ENV}=device=<UUIDs> naming {gpu_count} GPU UUIDs"
        )
    devices = _selector_devices(raw_selector)
    if len(devices) != gpu_count:
        raise GpuSelectorError(f"{label} but {GPU_SELECTOR_ENV} exposes {len(devices)} GPUs")
    if any(_STABLE_GPU_DEVICE_RE.fullmatch(device) is None for device in devices):
        raise GpuSelectorError("AGENT_CONTAINER_GPUS must name stable GPU UUID devices")
    canonical = {device.casefold() for device in devices}
    if len(canonical) != len(devices):
        raise GpuSelectorError("AGENT_CONTAINER_GPUS contains duplicate GPU UUIDs")
    return devices


def discover_gpu_selector(
    gpu_count: int,
    *,
    nvidia_smi: str = "nvidia-smi",
    timeout_seconds: float = 30.0,
    label: str | None = None,
) -> str:
    """Build a Docker selector for the first *gpu_count* GPUs ``nvidia-smi`` reports.

    Used when no explicit ``AGENT_CONTAINER_GPUS`` selector is set; the visible GPUs
    are assumed to belong to the caller (a cgroup-isolated job or a workstation).
    *label* names the requirement in diagnostics (default ``container.gpu_count=<n>``).
    """
    if type(gpu_count) is not int or gpu_count < 1:
        raise GpuSelectorError("GPU discovery requires a positive GPU count")
    label = label or f"container.gpu_count={gpu_count}"
    command = [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds, check=False
        )
    except FileNotFoundError:
        raise GpuSelectorError(f"{nvidia_smi} is not installed; set AGENT_CONTAINER_GPUS") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuSelectorError(f"could not run {nvidia_smi}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise GpuSelectorError(
            f"{nvidia_smi} exited with status {completed.returncode}: {detail[:200]}"
        )
    devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if any(_STABLE_GPU_DEVICE_RE.fullmatch(device) is None for device in devices):
        raise GpuSelectorError(f"{nvidia_smi} reported an unexpected GPU identifier")
    if len({device.casefold() for device in devices}) != len(devices):
        raise GpuSelectorError(f"{nvidia_smi} reported duplicate GPU UUIDs")
    if len(devices) < gpu_count:
        raise GpuSelectorError(f"{label} but {nvidia_smi} reports {len(devices)} GPUs")
    return bounded_docker_gpu_selector("device=" + ",".join(devices), gpu_count)


def bounded_docker_gpu_selector(
    raw_selector: str | None,
    gpu_count: int,
    *,
    offset: int = 0,
) -> str:
    """Select explicit devices and encode Docker's multi-device CSV value safely."""
    if gpu_count <= 0:
        raise GpuSelectorError("GPU count must be positive")
    if offset < 0:
        raise GpuSelectorError("GPU selector offset must not be negative")
    devices = _selector_devices(raw_selector)
    required = offset + gpu_count
    if len(devices) < required:
        raise GpuSelectorError(
            f"requires {required} GPUs but selector exposes {len(devices)}"
        )
    selected = devices[offset:required]
    bounded = "device=" + ",".join(selected)
    return f'"{bounded}"' if len(selected) > 1 else bounded


def gpu_pool_size(gpu_count: int, workers: int) -> int:
    """GPUs a run needs: *gpu_count* per container for each of *workers* parallel workers."""
    if type(gpu_count) is not int or gpu_count < 0:
        raise GpuSelectorError("GPU count must be a non-negative integer")
    if type(workers) is not int or workers < 1:
        raise GpuSelectorError("worker count must be a positive integer")
    return gpu_count * workers


def gpu_pool_label(gpu_count: int, workers: int) -> str:
    """Name the GPU requirement in diagnostics, with the worker factor when it applies."""
    if workers == 1:
        return f"container.gpu_count={gpu_count}"
    return f"container.gpu_count={gpu_count} x {workers} workers ({gpu_count * workers} GPUs)"


def settle_gpu_pool(
    raw_selector: str | None,
    gpu_count: int,
    workers: int,
    *,
    discover: Callable[..., str] = discover_gpu_selector,
) -> tuple[str, str] | None:
    """Validate or discover a run's GPU pool; returns ``(pool, first_slice)``.

    The pool names exactly ``gpu_count * workers`` stable GPU UUIDs: an explicit
    *raw_selector* must match that count, and without one *discover* takes the first
    that many visible GPUs. Each container receives ``gpu_count`` of them; the first
    slice is the allocation for everything that is not a parallel worker. ``None``
    means a CPU run, which requires an unset selector.
    """
    pool_size = gpu_pool_size(gpu_count, workers)
    label = gpu_pool_label(gpu_count, workers)
    if pool_size == 0:
        validate_gpu_allocation(raw_selector, 0, label=label)
        return None
    if raw_selector is None or not raw_selector.strip():
        raw_selector = discover(pool_size, label=label)
    validate_gpu_allocation(raw_selector, pool_size, label=label)
    pool = bounded_docker_gpu_selector(raw_selector, pool_size)
    return pool, bounded_docker_gpu_selector(pool, gpu_count)


def worker_gpu_selector(pool: str | None, gpu_count: int, worker_index: int) -> str:
    """The *gpu_count*-wide slice of *pool* owned by 1-based parallel *worker_index*."""
    if type(worker_index) is not int or worker_index < 1:
        raise GpuSelectorError("parallel worker index must be a positive integer")
    return bounded_docker_gpu_selector(pool, gpu_count, offset=(worker_index - 1) * gpu_count)
