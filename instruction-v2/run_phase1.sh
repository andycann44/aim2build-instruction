#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /absolute/or/relative/path/to/instruction.pdf [--run-id RUN_ID]"
  exit 1
fi

python3 "$SCRIPT_DIR/pipeline.py" phase1 --pdf "$1" "${@:2}"
