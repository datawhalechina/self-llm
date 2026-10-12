"""Start a tutorial script or module with torchada loaded before its imports."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="store_true", help="run a module, e.g. torch.distributed.run")
    parser.add_argument("target", help="Python script path or module name")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.module and not Path(args.target).is_file():
        parser.error(f"script does not exist: {args.target}")
    env = os.environ.copy()
    bootstrap = str(Path(__file__).resolve().parent / "bootstrap")
    env["SELF_LLM_MUSA"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [bootstrap, env.get("PYTHONPATH")]))
    trailing = args.args[1:] if args.args[:1] == ["--"] else args.args
    command = [sys.executable]
    if args.module:
        command.append("-m")
    command.extend([args.target, *trailing])
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
