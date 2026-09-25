"""Module entry point used by run-scoped backend integrations."""

from .app import main


if __name__ == "__main__":
    raise SystemExit(main())
