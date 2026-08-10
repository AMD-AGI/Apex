"""E2E resume dispatch with current campaign-baseline revalidation."""

from __future__ import annotations

import json

from apex.core import ApexError

from .formal import status_exit_code
from .release import require_campaign_baseline


def run_resume(args, build_application) -> int:
    baseline = require_campaign_baseline(args.release_candidate_receipt)
    application = build_application(include_e2e=True)
    if application.e2e_optimizer is None:
        raise ApexError("E2E composition is unavailable", "e2e_not_composed")
    result = application.e2e_optimizer.resume(
        args.run.expanduser().resolve(strict=True),
        campaign_baseline=baseline,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return status_exit_code(result.status)


__all__ = ["run_resume"]
