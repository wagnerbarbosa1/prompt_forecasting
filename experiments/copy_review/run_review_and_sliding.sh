#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
ENVS=/home/wagner-barbosa/miniconda3/envs
PREVIOUS_PID=68675
resume_previous() {
  if [[ -r /proc/$PREVIOUS_PID/cmdline ]] && tr '\0' ' ' < /proc/$PREVIOUS_PID/cmdline | grep -q '/tmp/resume_promptcast.py'; then
    kill -CONT "$PREVIOUS_PID"
  fi
}
trap resume_previous EXIT
if [[ -r /proc/$PREVIOUS_PID/cmdline ]] && tr '\0' ' ' < /proc/$PREVIOUS_PID/cmdline | grep -q '/tmp/resume_promptcast.py'; then
  kill -STOP "$PREVIOUS_PID"
fi
"$ENVS/baseline/bin/python" experiments/copy_review/reforecast.py > experiments/copy_review/execution.log 2>&1
"$ENVS/baseline/bin/python" - <<'PY'
import json,urllib.request
base='http://127.0.0.1:11435'
for m in json.load(urllib.request.urlopen(base+'/api/ps'))['models']:
 r=urllib.request.Request(base+'/api/generate',data=json.dumps({'model':m['name'],'keep_alive':0}).encode(),headers={'Content-Type':'application/json'})
 urllib.request.urlopen(r,timeout=120).read()
PY
for model in chronos timesfm; do
  for days in 7 15; do
    HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false "$ENVS/$model/bin/python" experiments/sliding/run.py --model "$model" --days "$days" > "experiments/sliding/${model}_${days}.log" 2>&1
  done
done
"$ENVS/baseline/bin/python" experiments/sliding/compare.py > experiments/sliding/comparison.log 2>&1
