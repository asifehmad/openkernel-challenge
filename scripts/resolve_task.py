"""Print a single absolute task directory; never treat layout discovery as provenance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.bundle import require_matching_contract, resolve_bundle  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--check-contract", action="store_true")
    args = p.parse_args()
    try:
        directory = resolve_bundle(args.path)
        if args.check_contract:
            require_matching_contract(directory, ROOT / "muse_bf16_proj")
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(directory)


if __name__ == "__main__":
    main()
