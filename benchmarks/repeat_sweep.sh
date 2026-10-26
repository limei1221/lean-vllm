#!/bin/bash
# Repeated serving curves for lean-vLLM and vLLM, alternating which engine goes first, as
# docs/2026-10-19-performance-optimization.md step 1 asks. Settings follow docs/benchmark-runbook.md.
#
#   benchmarks/repeat_sweep.sh OUT_DIR MODEL "RATES" [REPS] [ENGINES]
#
# Load 1 runs 200 requests rather than 1,000: it measures per-token latency, not a plateau, and 1,000 would take
# 17 minutes a run. Extra lean-vLLM serve flags go in $LEAN_EXTRA; extra environment for both servers is inherited.
set -u
out=$1 model=$2 rates=$3 reps=${4:-3} engines=${5:-lean-vllm,vllm}
cd "$(dirname "$0")/.."
KVTOKENS=${KVTOKENS:-327680}
PINNED="--max-num-batched-tokens ${BUDGET:-8192} --max-num-seqs 256 --enable-chunked-prefill"
TRACE_ARGS="--dataset lognormal --input-len 512 --output-len 128 --sigma 0.8 --temperature 0 --warmup 3 --timeout 1200"
SLO_ARGS="--slo-ttft ${SLO_TTFT:-1.0} --slo-tpot ${SLO_TPOT:-0.05}"
LEAN_SERVER_ARGS="$PINNED --cudagraph-mode full_and_piecewise --async-scheduling ${LEAN_EXTRA:-}"
VLLM_SERVER_ARGS="$PINNED --async-scheduling --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' ${VLLM_EXTRA:-}"
VPATH="$HOME/workspace/vllm-env/bin:$PATH"

mkdir -p "$out"
{
  git rev-parse HEAD; git status --short; git -C ~/workspace/FlashMLA rev-parse HEAD
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  uname -r; uv run --no-sync python -c 'import torch; print("lean-vLLM torch:", torch.__version__)'
  ~/workspace/vllm-env/bin/vllm --version
  echo "model=$model rates=$rates reps=$reps engines=$engines budget=${BUDGET:-8192} kvtokens=$KVTOKENS"
  echo "lean args: $LEAN_SERVER_ARGS"; echo "vllm args: $VLLM_SERVER_ARGS"; echo "slo: $SLO_ARGS"
  env | grep -E '^(LEAN_|VLLM_)' || true
  uptime; date -u '+%Y-%m-%dT%H:%M:%SZ'
} > "$out/environment.txt" 2>&1
git diff HEAD -- . ':!uv.lock' > "$out/working-tree.patch"
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -l 1 > "$out/nvidia-smi.log" &
gpu_log=$!
uptime_log() { while true; do echo "$(date -u +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done; }
uptime_log > "$out/loadavg.log" &
load_log=$!
trap 'kill $gpu_log $load_log 2>/dev/null' EXIT

sweep() {  # engine rates num_requests dir
  local engine=$1 rates=$2 n=$3 dir=$4 sargs=$LEAN_SERVER_ARGS path=$PATH
  if [ "$engine" = vllm ]; then sargs=$VLLM_SERVER_ARGS; path=$VPATH; fi
  echo "=== $(date -u +%T) $engine $rates x$n -> $dir"
  PATH="$path" uv run --no-sync python benchmarks/sweep.py --model "$model" --engine "$engine" --suite rate \
    --rates "$rates" --num-requests "$n" --seed 0 --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
    --server-args "$sargs" --client-args "$TRACE_ARGS $SLO_ARGS" --out "$dir" | tail -n +2
}

high=$(echo "$rates" | tr , '\n' | grep -vx 1 | paste -sd,)
IFS=, read -r -a order <<< "$engines"
for rep in $(seq 1 "$reps"); do
  if [ $((rep % 2)) -eq 0 ] && [ ${#order[@]} -gt 1 ]; then run_order=("${order[1]}" "${order[0]}"); else run_order=("${order[@]}"); fi
  for engine in "${run_order[@]}"; do
    case ",$rates," in *,1,*) sweep "$engine" 1 200 "$out/$engine/rep$rep-load1" ;; esac
    [ -n "$high" ] && sweep "$engine" "$high" 1000 "$out/$engine/rep$rep"
  done
done
echo "=== $(date -u +%T) done"
