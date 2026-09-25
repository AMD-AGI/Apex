"""Read/export/verify deterministic showcase projections."""

from __future__ import annotations

import json
from pathlib import Path

from apex.core import ContractError
from apex.reporting import verify_showcase

from .projections import export_showcase_projection


def add_showcase_commands(commands) -> None:
    showcase = commands.add_parser(
        "showcase", help="Export or inspect canonical showcase projections"
    )
    actions = showcase.add_subparsers(dest="showcase_command", required=True)
    export = actions.add_parser("export", help="Export one canonical run")
    export.add_argument("--run-root", type=Path, required=True)
    export.add_argument("--run-id")
    export.add_argument("--id", dest="showcase_id", required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = actions.add_parser("verify", help="Verify one exported showcase")
    verify.add_argument("--path", type=Path, required=True)
    show = actions.add_parser("show", help="Verify and render one showcase")
    show.add_argument("--path", type=Path, required=True)
    listing = actions.add_parser("list", help="Verify and list a showcase directory")
    listing.add_argument("--root", type=Path, required=True)


def run_showcase_command(args) -> int:
    if args.showcase_command == "export":
        result = export_showcase_projection(
            args.run_root,
            args.output,
            showcase_id=args.showcase_id,
            run_id=args.run_id,
        )
    elif args.showcase_command == "verify":
        result = _verification_document(verify_showcase(args.path))
    elif args.showcase_command == "show":
        verified = verify_showcase(args.path)
        result = json.loads((verified.root / "showcase.json").read_text(encoding="utf-8"))
    else:
        result = _list_showcases(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _list_showcases(root: Path) -> dict[str, object]:
    selected = root.expanduser()
    if selected.is_symlink():
        raise ContractError("Showcase list root cannot be a symlink", "unsafe_showcase_path")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise ContractError("Showcase list root does not exist", "showcase_missing") from error
    if not resolved.is_dir():
        raise ContractError("Showcase list root is not a directory", "invalid_showcase")
    entries = []
    for path in sorted(resolved.iterdir()):
        if path.is_dir() and not path.is_symlink() and (path / "showcase.json").is_file():
            entries.append(_verification_document(verify_showcase(path)))
    return {"schema": "apex.showcase-list/v1", "root": str(resolved), "entries": entries}


def _verification_document(value) -> dict[str, object]:
    return {
        **value.to_receipt(),
        "root": str(value.root),
    }


__all__ = ["add_showcase_commands", "run_showcase_command"]
