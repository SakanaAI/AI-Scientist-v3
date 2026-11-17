"""Detect evaluators whose score changes when the same program is evaluated twice."""
from __future__ import annotations

import math
from dataclasses import dataclass

# Tolerates float summation-order noise only; any real stochastic spread fails the replay.
REPLAY_RELATIVE_TOLERANCE = 1e-9
REPLAY_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    combined_score: float
    correct: bool


def replay_mismatch(first: EvaluationOutcome, replay: EvaluationOutcome) -> str | None:
    """Return a repair diagnostic when two evaluations of one program disagree."""
    scores_agree = math.isclose(
        first.combined_score,
        replay.combined_score,
        rel_tol=REPLAY_RELATIVE_TOLERANCE,
        abs_tol=REPLAY_ABSOLUTE_TOLERANCE,
    )
    if scores_agree and first.correct == replay.correct:
        return None
    return (
        "evaluator is not deterministic: evaluating the unchanged baseline twice gave "
        f"combined_score {first.combined_score!r} vs {replay.combined_score!r} and "
        f"correct {first.correct} vs {replay.correct}. The evaluator must return the same "
        "result for a fixed program. Seed every random stream from task-owned constants; a "
        "secret that hides seeds from the candidate must be fixed per task, never drawn per "
        "invocation (no secrets.*, os.urandom, uuid4, time-based, or unseeded RNG)."
    )
