"""Worker entry point for the locally fine-tuned multi-event model.

Provisional review signals are not final findings. In particular, a context-only
source must be disclosed as context, never rewritten as a statutory prohibition.
This adapter is not enabled merely by training a checkpoint.
"""
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

from backend.app.domain.schemas import Candidate
from ml.event_schema import parse
from ml.rules import RULES

ROOT = Path(__file__).resolve().parents[1]
TITLES = {'V1900019680': 'Правила кассовых операций и инкассации, постановление НБРК от 29.11.2019 №231',
          'V2000020075': 'Правила организации охраны и устройства помещений, постановление НБРК от 24.02.2020 №14'}


def path_setting(name, default):
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else ROOT/path


@lru_cache(maxsize=1)
def load_detector():
    model = path_setting('LOCAL_EVENT_MODEL_PATH', 'artifacts/models/qwen3-vl-2b-4bit')
    adapter = path_setting('LOCAL_EVENT_ADAPTER_PATH', 'artifacts/models/event-lora-all-v1')
    if not all(p.is_file() for p in [model/'config.json', adapter/'adapters.safetensors', adapter/'adapter_config.json']):
        raise RuntimeError('model_assets_missing')
    if sys.platform != 'darwin' or sys.version_info < (3, 10):
        raise RuntimeError('model_runtime_unavailable')
    from ml.multi_event import EventDetector
    return EventDetector(model, adapter)


def candidates_from_events(result):
    times = result.get('sampled_timestamps')
    duration = result.get('duration_sec')
    if (not isinstance(times, list) or not times or type(duration) not in [int, float]
            or not math.isfinite(duration) or duration <= 0
            or any(type(t) not in [int, float] or not math.isfinite(t) or not 0 <= t < duration for t in times)
            or times != sorted(set(times))):
        raise ValueError('invalid_model_output')
    import json
    decision = parse(json.dumps({k:result.get(k) for k in ['verdict', 'events']}), len(times))
    candidates = []
    for event in decision['events']:
        rule = RULES[event['category']]
        index = event['frame']
        evidence = times[max(0, index-1):min(len(times), index+2)]
        context = rule.relation == 'context_only'
        label = ('Событие для проверки: ' if context else 'Возможное событие: ')+rule.title
        explanation = ('Прямой запрет этого действия в двух исследованных документах не подтверждён. '
                       if context else 'Применимость требования зависит от подтверждения обстоятельств. ')
        candidates.append(Candidate(
            type=label, timecode_sec=times[index],
            description=('Модель отметила в последовательности кадров: '+rule.title.lower()+'. '
                         +explanation+'Необходимо проверить: '+rule.required_evidence),
            rule=rule.explanation+' '+rule.exclusions,
            source_title=TITLES[rule.source_id], source_url=rule.url,
            source_section='Пункты '+rule.section+(' — общий контекст, не прямой запрет' if context else ' — проверить условия применения'),
            evidence_timestamps=evidence))
    return candidates


def analyze(video_path):
    # The only runtime input is the uploaded video. No review/manifest/hash lookup.
    try:
        result = load_detector().analyze(Path(video_path))
    except TimeoutError:
        raise RuntimeError('analysis_timeout') from None
    return candidates_from_events(result)
