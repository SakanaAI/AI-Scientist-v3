"""Best-effort, metadata-only accounting for automated review API calls."""

from __future__ import annotations

import os
import sys
import uuid
from decimal import Decimal
from pathlib import Path

from .cost_tracking import ApiTokenPrice


# OpenAI standard-tier list-price snapshot checked 2026-11-20 (prompts <=272K input tokens):
# https://developers.openai.com/api/docs/models/gpt-5.6-terra
_GPT_5_6_TERRA_PRICE = ApiTokenPrice(
    input_per_million=Decimal("2.00"),
    cached_input_per_million=Decimal("0.20"),
    output_per_million=Decimal("12.00"),
    pricing_source="openai",
    pricing_status="pinned",
)
_REVIEW_TOKEN_PRICES = {
    "gpt-5.6-terra": _GPT_5_6_TERRA_PRICE,
}


def record_review_api_cost(
    response: object | None,
    *,
    stage: str,
    model: str,
    failed: bool = False,
) -> None:
    root = os.environ.get("AGENTIC_COST_ROOT")
    if not root:
        return
    try:
        from .cost_tracking import record_api_failure, record_api_response

        invocation_id = f"aireview:{stage}:{uuid.uuid4().hex}"
        if failed:
            record_api_failure(
                Path(root),
                phase="review",
                stage=stage,
                invocation_id=invocation_id,
                source="openai",
                model=model,
            )
        else:
            record_api_response(
                Path(root),
                response,
                phase="review",
                stage=stage,
                invocation_id=invocation_id,
                source="openai",
                token_prices=_REVIEW_TOKEN_PRICES,
            )
    except Exception as exc:
        print(f"[cost] warning: accounting unavailable ({type(exc).__name__})", file=sys.stderr)
