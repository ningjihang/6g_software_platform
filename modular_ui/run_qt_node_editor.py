from __future__ import annotations

import sys


def main() -> int:
    try:
        from .qt_node_editor import run_qt_node_workbench
    except Exception as exc:
        print("Failed to start Qt node workbench.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "The Qt runtime appears unavailable or incomplete in the current Python environment. "
            "Please repair the local Qt binding/runtime before launching the interface.",
            file=sys.stderr,
        )
        return 1
    return run_qt_node_workbench()


if __name__ == "__main__":
    raise SystemExit(main())
