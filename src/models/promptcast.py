"""Qwen forecasting through Ollama, with strict output validation and audit data."""
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OllamaConfig:
    model: str = 'qwen3:1.7b'
    base_url: str = 'http://127.0.0.1:11434'
    temperature: float = 0
    num_ctx: int = 8192
    num_predict: int = 256
    timeout: int = 600
    retries: int = 2
    seed: int = 42

    def __post_init__(self):
        if min(self.num_ctx, self.num_predict, self.timeout) < 1 or self.retries < 0:
            raise ValueError('Invalid Ollama limits')


def prediction_schema(horizon: int) -> dict:
    return {'type': 'object', 'properties': {'predictions': {
        'type': 'array', 'items': {'type': 'number', 'minimum': 0},
        'minItems': horizon, 'maxItems': horizon}},
        'required': ['predictions'], 'additionalProperties': False}


def parse_prediction(text: str, horizon: int) -> list[float]:
    """Reject prose, strings, booleans, NaN, missing values and wrong horizons."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {'predictions'}:
        raise ValueError('Expected exactly the predictions key')
    values = parsed['predictions']
    if not isinstance(values, list) or len(values) != horizon:
        raise ValueError(f'Expected {horizon} predictions')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Predictions must be finite, nonnegative numbers')
    return [float(v) for v in values]


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class QwenForecaster:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.last_audit = {}
        with urllib.request.urlopen(config.base_url + '/api/tags', timeout=10) as response:
            tags = json.load(response)
        matches = [m for m in tags['models'] if m['name'] == config.model]
        if not matches:
            raise ValueError(f'Model {config.model} is not installed in Ollama')
        metadata = post_json(config.base_url + '/api/show', {'model': config.model}, 30)
        info = metadata['model_info']
        architecture = info['general.architecture']
        maximum_context = int(info[f'{architecture}.context_length'])
        if config.num_ctx > maximum_context:
            raise ValueError(f'Requested {config.num_ctx} tokens exceeds model limit {maximum_context}')
        self.identity = {'model': config.model, 'digest': matches[0]['digest'],
                         'maximum_context_tokens': maximum_context, 'config': asdict(config)}

    def predict(self, history, horizon: int, *, prompt: str, future_covariates=None) -> list[float]:
        """Send only a prebuilt, target-free prompt; retain every attempted response."""
        config = self.config
        # UTF-8 byte count is a conservative upper bound for byte-level tokenizers.
        # Reserve space for the Ollama model template and the entire generated answer.
        if len(prompt.encode('utf-8')) + config.num_predict + 512 > config.num_ctx:
            raise ValueError('Prompt exceeds conservative context budget; increase num_ctx')
        payload = {'model': config.model, 'prompt': prompt, 'stream': False,
                   'think': False, 'keep_alive': '10m', 'truncate': False, 'format': prediction_schema(horizon),
                   'options': {'temperature': config.temperature, 'seed': config.seed,
                               'num_ctx': config.num_ctx, 'num_predict': config.num_predict}}
        self.last_audit = {'request': payload, 'attempts': []}
        for attempt in range(config.retries + 1):
            audit = {'attempt': attempt + 1}
            self.last_audit['attempts'].append(audit)
            try:
                answer = post_json(config.base_url + '/api/generate', payload, config.timeout)
                audit['response'] = answer
                if not answer.get('done') or answer.get('done_reason') == 'length':
                    raise ValueError('Incomplete or token-limited Ollama response')
                if answer.get('prompt_eval_count', 0) + answer.get('eval_count', 0) >= config.num_ctx:
                    raise ValueError('Context limit reached')
                return parse_prediction(answer.get('response', ''), horizon)
            except (ValueError, urllib.error.URLError, TimeoutError) as exc:
                audit['error'] = str(exc)
                if attempt == config.retries:
                    raise
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError('Unreachable retry state')
