#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-/root/autodl-fs/experiments/minimax-m3-8gpu}"
PATCH_FILE="$LAB_ROOT/patches/transformers-5.12.1-zero3-streaming.patch"

version="$(python -c 'import transformers; print(transformers.__version__)')"
if [ "$version" != "5.12.1" ]; then
  echo "expected transformers 5.12.1, found $version" >&2
  exit 1
fi

site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
target="$site_packages/transformers/modeling_utils.py"
backup="$target.minimax-m3.orig"

if patch --batch --forward --dry-run -p2 -d "$site_packages" < "$PATCH_FILE" >/dev/null; then
  cp -n "$target" "$backup"
  patch --batch --forward -p2 -d "$site_packages" < "$PATCH_FILE"
elif patch --batch --reverse --dry-run -p2 -d "$site_packages" < "$PATCH_FILE" >/dev/null; then
  echo "Transformers MiniMax-M3 ZeRO-3 patch is already applied."
else
  echo "patch does not match $target; reinstall transformers 5.12.1 and retry" >&2
  exit 1
fi
