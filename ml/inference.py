"""Worker adapter for the experimental local cash-category detector.

No filenames, organizer labels or annotated event times are inference inputs.
Uncertain analysis raises an explicit error rather than returning no_violation.
"""
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from backend.app.domain.schemas import Candidate

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'Возможное оставление банкнот после завершения обслуживания'
SOURCE_URL = 'https://adilet.zan.kz/rus/docs/V1900019680'
SOURCE_TITLE = ('Правила осуществления кассовых операций и операций по инкассации банкнот, '
                'монет и ценностей. Постановление Правления НБРК от 29.11.2019 № 231')
RULE = ('Пункт 20: ранее принятые наличные деньги хранятся в предназначенных для этого '
        'устройствах на рабочем месте, закрывающихся на ключ. '
        'Необходимо проверить, были ли деньги приняты и применимо ли требование к данному фрагменту.')


def local_path(variable, default):
    path = Path(os.getenv(variable, default)).expanduser()
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def load_detector():
    model = local_path('LOCAL_VLM_PATH', 'artifacts/models/qwen3-vl-4b-4bit')
    objects = local_path('LOCAL_CASH_MODEL_PATH', 'artifacts/models/dense-cash-v2')
    encoder = local_path('LOCAL_DINO_PATH', 'artifacts/models/dinov2-small-onnx')
    annotations = local_path('LOCAL_ANNOTATIONS_PATH', 'data/processed/manual-grounding-v1/annotations.json')
    required = [model/'config.json', objects/'classifier.npz', objects/'support.npz',
                encoder/'model.onnx', annotations]
    if not all(path.is_file() for path in required):
        raise RuntimeError('model_assets_missing')
    if sys.version_info < (3, 10) or sys.platform != 'darwin':
        raise RuntimeError('model_runtime_unavailable')
    try:
        from ml.dense_cash import DenseCashDetector
        return DenseCashDetector(model, annotations=annotations, crop_model=objects,
                                 dino_model=encoder, max_side=448, deadline_sec=230)
    except ImportError:
        raise RuntimeError('model_runtime_unavailable') from None


def result_to_candidates(result, scans):
    """Map detector output to API objects; validate the selected temporal evidence."""
    prediction = result.get('prediction')
    if prediction is None:
        raise RuntimeError('analysis_uncertain')
    if type(prediction) is not int or prediction not in [0, 1]:
        raise ValueError('invalid_model_output')
    if prediction == 0:
        if result.get('event_times_sec'):
            raise ValueError('invalid_model_output')
        return []
    times = scans.get('timestamps', [])
    observations, phases = scans.get('scans', []), scans.get('occupancy', [])
    duration = result.get('duration_sec')
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
            or len(times) != len(observations) or len(times) != len(phases) or len(times) < 2
            or any(type(t) not in [int, float] or not math.isfinite(t) or not 0 <= t < duration for t in times)
            or times != sorted(set(times))
            or any(type(p) is not int or p not in [0, 1, 2] for p in phases)
            or any(type(o.get('cash_detected')) is not bool for o in observations)):
        raise ValueError('invalid_model_output')
    confirmed = [i for i in range(len(times)-1)
                 if observations[i]['cash_detected'] and observations[i+1]['cash_detected']
                 and phases[i] == phases[i+1] == 1]
    if not confirmed:
        raise ValueError('invalid_model_output')
    # Consecutive confirmations describe one episode, not several independent events.
    starts = [i for j, i in enumerate(confirmed) if j == 0 or i > confirmed[j-1]+1]
    candidates = []
    for i in starts:
        evidence = times[max(0, i-1):i+2]
        candidates.append(Candidate(
            type='Возможное оставление банкнот после обслуживания', timecode_sec=times[i],
            description=('Модель отметила в последовательности кадров признаки банкнот на рабочей '
                         'поверхности и завершения обслуживания либо ухода сотрудника. '
                         'Это предварительный сигнал: проверьте предметы, соседние кадры и то, '
                         'относятся ли деньги к ранее принятым.'),
            rule=RULE, source_title=SOURCE_TITLE, source_url=SOURCE_URL,
            source_section='Пункт 20, абзац 2', evidence_timestamps=evidence))
    return candidates


def analyze(video_path):
    video_path = Path(video_path)
    detector = load_detector()
    log_dir = local_path('STORAGE_ROOT', 'data/runtime')/'ml_logs'/str(uuid4())
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = detector.analyze(video_path, log_dir)
        scans_path = log_dir/'scans.json'
        scans = json.loads(scans_path.read_text()) if scans_path.exists() else {}
        candidates = result_to_candidates(result, scans)
        (log_dir/'adapter.json').write_text(json.dumps({
            'scope': SCOPE, 'experimental': True, 'events': len(candidates),
            'uncertainty_policy': 'failed/analysis_uncertain',
            'source_url': SOURCE_URL, 'source_section': '20'}, ensure_ascii=False, indent=2), encoding='utf-8')
        return candidates
    except TimeoutError:
        raise RuntimeError('analysis_timeout') from None
