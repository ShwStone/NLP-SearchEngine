#!/usr/bin/env bash
# Start the three configured serving processes, gateway and optional SSH tunnel.
# Qdrant instances are intentionally managed separately from their snapshots.
set -euo pipefail
RUNTIME="${1:?usage: deploy/start_local.sh /absolute/runtime [python]}"
PYTHON="${2:-python3}"
ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$RUNTIME/logs"
export PYTHONPATH="$RUNTIME/python-deps:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

start_process() {
  local name="$1" module="$2" config="$3"
  local pidfile="$RUNTIME/logs/$name.pid" logfile="$RUNTIME/logs/$name.log"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    if ps -p "$(cat "$pidfile")" -o args= | grep -F -- "$config" >/dev/null; then
      echo "$name already running with pid $(cat "$pidfile")"
      return
    fi
    echo "$name pid file points to an unrelated live process" >&2
    return 1
  fi
  (
    cd "$RUNTIME"
    nohup "$PYTHON" -B -u -m "$module" --config "$config" >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  echo "$name started with pid $(cat "$pidfile")"
}

start_process news20k online.server "$RUNTIME/config/news20k.json"
start_process mini100k online.server "$RUNTIME/config/mini100k.json"
start_process official8m online.server "$RUNTIME/config/official8m.json"
start_process gateway online.gateway "$RUNTIME/config/datasets.json"

if [[ "${START_TUNNEL:-1}" == 1 ]]; then
  pidfile="$RUNTIME/logs/tunnel.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "tunnel already running with pid $(cat "$pidfile")"
  else
    nohup ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R 127.0.0.1:18081:127.0.0.1:18083 Aliyun >>"$RUNTIME/logs/tunnel.log" 2>&1 &
    echo $! >"$pidfile"
    echo "tunnel started with pid $(cat "$pidfile")"
  fi
fi
