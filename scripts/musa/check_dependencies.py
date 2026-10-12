"""Reject new dependency conflicts while preserving the vendor image baseline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def conflicts():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True, capture_output=True,
    )
    # pip may report metadata parse errors on stderr while stdout only lists
    # old conflicts. That is an incomplete check, not an unchanged baseline.
    if (result.returncode not in (0, 1) or result.stderr.strip()
            or (result.returncode == 1 and not result.stdout.strip())):
        raise RuntimeError(f"pip check failed: {result.stderr or result.stdout}")
    if result.returncode == 0:
        return []
    return sorted(set(line for line in result.stdout.splitlines() if line.strip()))


def compare(baseline, current):
    return sorted(set(current) - set(baseline))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "compare"])
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    current = conflicts()
    if args.action == "snapshot":
        args.baseline.write_text(json.dumps(current), encoding="utf-8")
        if current:
            print("Image dependency conflicts already present before installation:")
            print("\n".join(current))
        print(json.dumps({"baseline_conflicts": len(current)}))
        return 0
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not isinstance(baseline, list) or not all(isinstance(line, str) for line in baseline):
        raise ValueError("Invalid dependency baseline")
    added = compare(baseline, current)
    print(json.dumps({"baseline_conflicts": len(baseline), "current_conflicts": len(current),
                      "new_conflicts": added}))
    return 1 if added else 0


if __name__ == "__main__":
    sys.exit(main())
