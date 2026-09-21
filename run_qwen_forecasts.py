#!/usr/bin/env python3
"""Run existing PromptCast prompts against local Ollama; resume saved successes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent

def parse_prediction(text, horizon):
    text = text.strip()
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    if not re.fullmatch(number + r'(?:\s*,\s*' + number + r')*', text):
        raise ValueError('Response is not exclusively comma-separated numbers')
    values = [float(x) for x in text.split(',')]
    if len(values) != horizon or not all(math.isfinite(x) for x in values):
        raise ValueError(f'Expected {horizon} finite predictions; received {len(values)}')
    return values

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3:1.7b')
    parser.add_argument('--prompts', type=Path, default=ROOT / 'initial_prompts/promptcast_prompts')
    parser.add_argument('--output', type=Path, default=ROOT / 'output_models/qwen3_1_7b')
    parser.add_argument('--limit', type=int, default=0, help='Maximum pending prompts; 0 runs all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.prompts / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    options = {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 256}
    pending = []
    for row in rows:
        prompt = (args.prompts / row['prompt_file']).read_text()
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        target = args.output / (row['prompt_id'] + '.json')
        if target.exists():
            saved = json.loads(target.read_text())
            if (saved.get('status') == 'ok' and saved.get('prompt_sha256') == digest
                    and saved.get('model') == args.model and saved.get('options') == options):
                continue
        pending.append((row, prompt, digest, target))
    if args.limit:
        pending = pending[:args.limit]
    print(f'{len(rows)} prompts in manifest; {len(pending)} pending in this run', flush=True)
    failures = 0
    for i, (row, prompt, digest, target) in enumerate(pending, 1):
        start = time.monotonic()
        result = dict(row, model=args.model, options=options, think=False,
                      prompt_sha256=digest, status='error')
        payload = dict(model=args.model, prompt=prompt, stream=False, think=False, options=options)
        request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
            result['raw_response'] = answer.get('response', '')
            result['done_reason'] = answer.get('done_reason')
            if not answer.get('done') or answer.get('done_reason') == 'length':
                raise ValueError('Incomplete model response')
            result['predictions'] = parse_prediction(result['raw_response'], int(row['horizon_days']))
            result['status'] = 'ok'
        except Exception as exc:
            result['error'] = str(exc)
            failures += 1
        result['elapsed_seconds'] = round(time.monotonic() - start, 3)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(target)
        print(f"[{i}/{len(pending)}] {row['prompt_id']}: {result['status']} ({result['elapsed_seconds']}s)", flush=True)
        if 'error' in result:
            print(result['error'], flush=True)
            if 'raw_response' not in result:
                raise SystemExit('Ollama request failed; fix the server and rerun to resume.')
    print(f'Finished: {len(pending) - failures} successful, {failures} failed. Output: {args.output}', flush=True)
    raise SystemExit(1 if failures else 0)

if __name__ == '__main__':
    main()
