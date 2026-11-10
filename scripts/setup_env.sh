#!/usr/bin/env bash
# Tao moi truong ao .venv va cai DLNet (Linux / macOS / Git Bash).
#
#   ./scripts/setup_env.sh            # CPU torch (nhe, chay tren moi may)
#   ./scripts/setup_env.sh --gpu      # torch mac dinh cua PyPI (co CUDA tren Linux/Windows)
#   ./scripts/setup_env.sh --no-check # bo buoc tu kiem tra cuoi
#
# Sau do:  source .venv/bin/activate   roi   dlnet selfcheck
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" # "" = PyPI mac dinh
CHECK=1
for a in "$@"; do
  case "$a" in
    --gpu) TORCH_INDEX="" ;;
    --no-check) CHECK=0 ;;
    -h | --help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "tham so la: $a (xem --help)" >&2; exit 2 ;;
  esac
done
[ "$(uname -s)" = "Darwin" ] && TORCH_INDEX="" # macOS: wheel PyPI da la ban CPU/MPS

PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)'; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Can Python 3.10 - 3.13 (khong tim thay)." >&2; exit 1; }
echo "[env] dung $($PY --version) -> $VENV"

"$PY" -m venv "$VENV" || { echo "Tao venv that bai. Debian/Ubuntu: sudo apt install python3-venv" >&2; exit 1; }
if [ -x "$VENV/bin/python" ]; then VPY="$VENV/bin/python"; else VPY="$VENV/Scripts/python.exe"; fi
"$VPY" -m pip install --quiet --upgrade pip

if [ -n "$TORCH_INDEX" ]; then
  echo "[env] cai torch (CPU) tu $TORCH_INDEX"
  "$VPY" -m pip install torch --index-url "$TORCH_INDEX" || {
    echo "[env] khong dung duoc $TORCH_INDEX, thu torch tu PyPI (nang hon)" >&2
    "$VPY" -m pip install torch
  }
else
  echo "[env] cai torch tu PyPI"
  "$VPY" -m pip install torch
fi
echo "[env] cai DLNet va cac thu vien con lai"
"$VPY" -m pip install -e ".[dev,design]"

if [ "$CHECK" = 1 ]; then
  echo "[env] tu kiem tra"
  "$VPY" -m dlnet selfcheck
fi
cat <<MSG

Xong. Kich hoat moi truong:
  Linux/macOS:  source $VENV/bin/activate
  Git Bash:     source $VENV/Scripts/activate
Roi chay thu:   dlnet selfcheck
MSG
