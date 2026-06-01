"""``python -m online_emotion [run|export|serve] ...`` dispatcher."""
from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "run"
    rest = argv[1:] if argv else []
    if cmd == "run":
        from .cli.run import main as run_main

        return run_main(rest)
    if cmd == "export":
        from .cli.export import main as export_main

        return export_main(rest)
    if cmd == "serve":
        from .serve import main as serve_main

        return serve_main(rest)
    print("usage: python -m online_emotion [run|export|serve] ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
