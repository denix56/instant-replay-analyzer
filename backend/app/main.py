from __future__ import annotations

from .api.server import create_app

app = create_app()


def main() -> int:
    from .cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
