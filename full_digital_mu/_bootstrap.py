import sys
from pathlib import Path


def ensure_classical_on_path() -> Path:
    """??classical on path?"""
    repo_root = Path(__file__).resolve().parents[1]
    classical_dir = repo_root / "classical"

    for path in (repo_root, classical_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return classical_dir
