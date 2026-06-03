#!/usr/bin/env bash
# Prepare all three datasets sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_uicvd.py
python scripts/prepare_gengui.py
python scripts/prepare_vins.py "$@"
