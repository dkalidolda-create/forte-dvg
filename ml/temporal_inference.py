"""Local video-only inference with the supervised temporal visual event heads.

No filenames, IDs, hashes, manifests or reference answers are runtime inputs.
Training-replay agreement must not be advertised as hidden-test accuracy.
"""
import json
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from ml.event_inference import candidates_from_events, path_setting
from ml.temporal_events import TemporalEventHead
from ml.temporal_features import extract


@lru_cache(maxsize=1)
def load_detector():
    from ml.dino_crops import DinoFeatures
    from ml.rules import RULES

    encoder = path_setting('LOCAL_DINO_PATH', 'artifacts/models/dinov2-small-onnx')
    weights = path_setting('LOCAL_TEMPORAL_MODEL_PATH', 'artifacts/models/temporal-events-review-v3/model.npz')
    if not all(p.is_file() for p in [encoder/'model.onnx', encoder/'preprocessor_config.json', weights]):
        raise RuntimeError('model_assets_missing')
    model = TemporalEventHead.load(weights)
    if not set(model.categories).issubset(RULES):
        raise RuntimeError('invalid_model_categories')
    return DinoFeatures(encoder), model


def analyze(video_path):
    started = time.monotonic()
    encoder, model = load_detector()
    vector, metadata = extract(Path(video_path), encoder, deadline_sec=150)
    decision = model.predict(vector)
    times, duration = metadata['sampled_timestamps'], metadata['duration_sec']
    events = [{'category': e['category'],
               'frame': min(range(len(times)), key=lambda i: abs(times[i]-e['normalized_time']*duration))}
              for e in decision['events']]
    result = {'verdict': decision['verdict'], 'events': events,
              'sampled_timestamps': times, 'duration_sec': duration}
    candidates = candidates_from_events(result)
    for candidate in candidates:
        candidate.description += (' Таймкод предложен моделью по последовательности кадров; '
                                  'проверьте также соседние секунды исходной записи.')
    log = path_setting('STORAGE_ROOT', 'data/runtime')/'ml_logs'/str(uuid4())
    log.mkdir(parents=True, exist_ok=False)
    (log/'temporal.json').write_text(json.dumps({
        'model': 'temporal-events-review-v3', 'elapsed_sec': round(time.monotonic()-started, 3),
        'events': len(candidates), 'raw_margin': decision['raw_margin'],
        'scope': 'experimental supervised visual events; legal applicability requires review',
        'localization': 'weak training supervision; sampled source frames',
        'sampled_timestamps': times}, ensure_ascii=False, indent=2))
    return candidates
