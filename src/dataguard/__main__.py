"""Allow running the CLI with python -m dataguard."""

from dataguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
