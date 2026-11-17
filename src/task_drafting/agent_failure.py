"""Classify bounded task-drafting agent failures for safe retries."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from enum import Enum
from pathlib import Path

_MAX_LOG_TAIL_BYTES = 64 * 1024
_HEADLESS_CAPACITY_PATTERN = re.compile(r"selected model is at capacity", re.IGNORECASE)
_TRANSIENT_PATTERNS = (
    re.compile(r"\b(?:http\s+)?429(?:\s*:|\s+too many requests\b)", re.IGNORECASE),
    re.compile(r"\brate limit(?:ed| exceeded)\b", re.IGNORECASE),
    re.compile(r"\bRESOURCE_EXHAUSTED\b", re.IGNORECASE),
    re.compile(r"\b(?:service|provider|upstream)\b.{0,80}\b(?:overloaded|at capacity)\b", re.IGNORECASE),
    re.compile(r"\b(?:provider|upstream)\b.{0,80}\b(?:502|503|504)\b", re.IGNORECASE),
    re.compile(r"\b(?:502|503|504)\b.{0,80}\bservice unavailable\b", re.IGNORECASE),
    re.compile(r"\bconnection (?:reset|closed) by peer\b", re.IGNORECASE),
    re.compile(r"\b(?:request|connection) timed out\b.{0,80}\b(?:model|provider|upstream)\b", re.IGNORECASE),
    re.compile(r"\b(?:model|provider|upstream)\b.{0,80}\brequest timed out\b", re.IGNORECASE),
    # Host network loss while the agent CLI talks to its provider (issue #2).
    re.compile(r"\b(?:ECONNRESET|ECONNREFUSED|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|ENETUNREACH|EHOSTUNREACH)\b"),
    re.compile(r"\bAPIConnectionError\b"),
    re.compile(r"\bAPI Error\b.{0,40}\b(?:connection error|request timed out)", re.IGNORECASE),
    re.compile(r"\btemporary failure in name resolution\b", re.IGNORECASE),
    re.compile(r"\bstream disconnected before completion\b", re.IGNORECASE),
    re.compile(r"\bsocket hang up\b", re.IGNORECASE),
    re.compile(r"\bTypeError: fetch failed\b"),
)
# Claude stream-json user/assistant turns and Codex item events carry tool output and prose.
_TRANSCRIPT_EVENT_TYPES = frozenset({"user", "assistant", "item.started", "item.updated", "item.completed"})
_PERMANENT_PATTERNS = (
    re.compile(r"\binvalid (?:api key|token|credentials)\b", re.IGNORECASE),
    re.compile(r"\b(?:authentication failed|unauthorized|forbidden)\b", re.IGNORECASE),
    re.compile(r"\bpermission denied\b", re.IGNORECASE),
    re.compile(r"\b(?:model|profile) (?:not found|does not exist)\b", re.IGNORECASE),
)


class AgentFailureKind(str, Enum):
    """Retry disposition for one failed agent invocation."""

    TRANSIENT_PROVIDER = "transient_provider"
    NON_RETRYABLE = "non_retryable"


def classify_agent_failure_text(text: str) -> AgentFailureKind:
    """Classify bounded agent output without returning its contents."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if not _is_transcript_event(line)][-32:]
    structured_error = _last_structured_error(lines)
    candidates = [structured_error] if structured_error is not None else reversed(lines)
    for candidate in candidates:
        # Headless already spent its native capacity retry budget (or retries were disabled).
        if _HEADLESS_CAPACITY_PATTERN.search(candidate):
            return AgentFailureKind.NON_RETRYABLE
        if any(pattern.search(candidate) for pattern in _PERMANENT_PATTERNS):
            return AgentFailureKind.NON_RETRYABLE
        if any(pattern.search(candidate) for pattern in _TRANSIENT_PATTERNS):
            return AgentFailureKind.TRANSIENT_PROVIDER
    return AgentFailureKind.NON_RETRYABLE


def _json_event(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_transcript_event(line: str) -> bool:
    """Agent prose and tool output, which may quote errors that are not the agent's own."""
    event = _json_event(line)
    return event is not None and event.get("type") in _TRANSCRIPT_EVENT_TYPES


def _last_structured_error(lines: list[str]) -> str | None:
    for line in reversed(lines):
        event = _json_event(line)
        if event is None or event.get("type") not in {"error", "turn.failed"}:
            continue
        message = event.get("message")
        error = event.get("error")
        if event.get("type") == "turn.failed" and isinstance(error, dict):
            message = error.get("message")
        if isinstance(message, str):
            return message
    return None


def classify_agent_failure_log(path: Path) -> AgentFailureKind:
    """Safely classify the bounded tail of one harness-owned agent log."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("agent failure log must be an owned regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("agent failure log must be an owned regular file")
        offset = max(0, metadata.st_size - _MAX_LOG_TAIL_BYTES)
        os.lseek(descriptor, offset, os.SEEK_SET)
        content = os.read(descriptor, _MAX_LOG_TAIL_BYTES)
    finally:
        os.close(descriptor)
    return classify_agent_failure_text(content.decode("utf-8", errors="replace"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify one task-drafting agent failure")
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    try:
        kind = classify_agent_failure_log(args.log)
    except ValueError:
        return 2
    return 0 if kind is AgentFailureKind.TRANSIENT_PROVIDER else 1


if __name__ == "__main__":
    raise SystemExit(main())
