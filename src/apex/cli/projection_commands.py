"""Thin CLI rendering for report and RL projection services."""

from __future__ import annotations

import json

from apex.rl import DatasetExportConfig

from .projections import export_rl_dataset, rebuild_report


def report_command(args) -> int:
    result = rebuild_report(args.run_root, args.output, run_id=args.run_id)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"reported {result['run_id']} to {args.output.expanduser().resolve()}")
    return 0


def export_rl_command(args) -> int:
    result = export_rl_dataset(
        args.run_root,
        args.output,
        run_id=args.run_id,
        config=DatasetExportConfig(
            split=args.split,
            policy_id=args.policy_id,
            on_incomplete=args.on_incomplete,
            include_sft=not args.no_sft,
        ),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"exported {result['record_count']} episodes "
            f"({result['sft_count']} SFT) to {result['output_dir']}"
        )
    return 0


__all__ = ["export_rl_command", "report_command"]
