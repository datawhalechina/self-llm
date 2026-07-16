#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="MiniMax/MiniMax-M3")
    parser.add_argument("--output", default="/root/autodl-tmp/models/MiniMax-M3-BF16")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output.parent).free / 1024**3
    if not output.exists() and free_gib < 850:
        raise RuntimeError(f"Need about 850 GiB free for a fresh download; found {free_gib:.1f} GiB")

    path = snapshot_download(
        args.model_id,
        local_dir=str(output),
        max_workers=args.workers,
    )
    print(path)


if __name__ == "__main__":
    main()
