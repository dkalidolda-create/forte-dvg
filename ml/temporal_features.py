"""Offline scene-motion features; no file identity, people identification or labels.

Frozen DINOv2 is used as a generic visual encoder. The fitted event heads are
experimental: scene classification does not prove a norm, object or timestamp.
"""
import time

import numpy as np

from ml.grounded_cash import choose_indexes
from ml.local_cash import sample_video
from ml.vision_crops import normalize


def aggregate(features):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError('Invalid temporal features')
    # Retain chronology rather than treating screenshots as independent samples.
    parts = [values[index].mean(axis=0).ravel() for index in np.array_split(np.arange(len(values)), 3)]
    parts += [(values[-1]-values[0]).ravel(), np.abs(np.diff(values, axis=0)).mean(axis=0).ravel()]
    return normalize(np.concatenate([normalize(part) for part in parts])).astype(np.float32)


def extract(video_path, encoder, deadline_sec=100):
    from PIL import Image

    started = time.monotonic()
    frames, times, duration = sample_video(video_path, 1, 848, started+deadline_sec)
    indexes = choose_indexes(len(frames), 12)
    vectors = []
    for index in indexes:
        if time.monotonic()-started > deadline_sec:
            raise TimeoutError('analysis_timeout')
        image = frames[index]
        w, h = image.size
        regions = [image] + [image.crop(box) for box in [
            (0, 0, w//2, h//2), (w//2, 0, w, h//2),
            (0, h//2, w//2, h), (w//2, h//2, w, h)]]
        # Fit the whole region into the encoder; the default centre crop could hide a door/safe.
        vectors.append([encoder.encode(region.resize((224, 224), Image.Resampling.BICUBIC))
                        for region in regions])
    raw = np.asarray(vectors)
    return aggregate(raw), {'frames': len(indexes), 'duration_sec': duration,
                            'sampled_timestamps': [times[i] for i in indexes],
                            'elapsed_sec': round(time.monotonic()-started, 3)}


def fit_binary(vectors, targets, alpha=1.0):
    vectors = normalize(np.asarray(vectors, dtype=np.float64))
    targets = np.asarray(targets, dtype=int)
    if set(targets.tolist()) != {0, 1}:
        raise ValueError('Both classes required')
    counts = np.bincount(targets, minlength=2)
    root = np.sqrt(len(targets)/(2*counts[targets]))
    kernel = vectors@vectors.T+1
    dual = root*np.linalg.solve(root[:, None]*kernel*root[None, :]+alpha*np.eye(len(targets)),
                                root*(targets*2-1))
    return vectors, dual


def binary_scores(vectors, support, dual):
    return (normalize(np.asarray(vectors))@support.T+1)@dual
