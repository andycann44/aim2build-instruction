#!/bin/bash
: "${HISTTIMEFORMAT:=}"
set -euo pipefail
[ -f ./a2p_bash_compat.sh ] && source ./a2p_bash_compat.sh
[ -f /tmp/a2p_env.sh ] && source /tmp/a2p_env.sh

cd /Users/olly/aim2build-instruction

PAGE="${1:-6}"

curl -s "http://127.0.0.1:8000/api/analyze?page=${PAGE}" | python3 - <<'PY'
import sys, json
j = json.load(sys.stdin)

for k in [
    "page",
    "panel_found",
    "panel_source",
    "panel_box",
    "shell_found",
    "shell_box",
    "grey_bag_found",
    "number_box_found",
    "bag_number",
    "confidence"
]:
    print(f"{k}: {j.get(k)}")
PY
