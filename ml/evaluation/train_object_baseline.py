"""Supervised crop classifier with source-video separation; NOT event detection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ['banknotes', 'documents', 'cash_storage']


def features(image):
    rgb = np.asarray(image.convert('RGB').resize((8, 8)), dtype=float)/255
    hsv = np.asarray(image.convert('HSV').resize((32, 32)), dtype=float)/255
    hist = [np.histogram(hsv[:, :, channel], bins=12, range=(0, 1), density=False)[0]/1024
            for channel in range(3)]
    gray = np.asarray(image.convert('L').resize((24, 24)), dtype=float)/255
    dy, dx = np.gradient(gray)
    angle, magnitude = np.arctan2(dy, dx), np.hypot(dx, dy)
    gradients = []
    for y in range(0, 24, 8):
        for x in range(0, 24, 8):
            values, _ = np.histogram(angle[y:y+8, x:x+8], bins=8, range=(-np.pi, np.pi),
                                     weights=magnitude[y:y+8, x:x+8])
            gradients.extend(values/(np.linalg.norm(values)+1e-6))
    return np.concatenate([rgb.ravel(), *hist, gradients])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--train-video', default='X-5')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    if data['split'] != 'train':
        raise ValueError('Use manual train annotations only')
    rows, vectors = [], []
    for frame in data['annotations']:
        for box in frame['boxes']:
            rows.append({'video_id': frame['video_id'], 'target': box['object'],
                         'timestamp_sec': frame['timestamp_sec']})
            vectors.append(features(Image.open(args.annotations.parent/box['crop'])))
    vectors = np.stack(vectors)
    targets = np.array([CLASSES.index(r['target']) for r in rows])
    train = np.array([r['video_id'] == args.train_video for r in rows])
    if set(targets[train]) != set(range(3)) or not (~train).any():
        raise ValueError('Training source must contain all classes; other source videos are required')
    mean, scale = vectors[train].mean(axis=0), vectors[train].std(axis=0)+.05
    normalized = (vectors-mean)/scale
    x = np.column_stack([normalized[train], np.ones(train.sum())])
    y = np.eye(3)[targets[train]]
    weights = np.zeros((x.shape[1], 3))
    sample_weights = train.sum()/(3*np.bincount(targets[train], minlength=3)[targets[train]])
    for _ in range(1200):
        logits = x@weights
        probabilities = np.exp(logits-logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        regularization = .1*weights
        regularization[-1] = 0
        gradient = x.T@((probabilities-y)*sample_weights[:, None])/len(x)+regularization
        weights -= .02*gradient
    test = np.column_stack([normalized[~train], np.ones((~train).sum())])
    predictions = (test@weights).argmax(axis=1)
    expected = targets[~train]
    confusion = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(expected, predictions):
        confusion[actual, predicted] += 1
    details = [{**r, 'prediction': CLASSES[int(p)]}
               for r, p in zip([r for r, t in zip(rows, train) if not t], predictions)]
    recalls = [float((predictions[expected == c] == c).mean()) if (expected == c).any() else None
               for c in range(3)]
    report = {'scope': 'Object crops only; source-held-out inside TRAIN. Not video/event accuracy.',
              'train_source_video': args.train_video, 'train_crops': int(train.sum()),
              'test_source_videos': sorted({r['video_id'] for r in details}),
              'test_crops': int((~train).sum()), 'accuracy': float((predictions == expected).mean()),
              'classes': CLASSES, 'confusion': confusion.tolist(), 'recall_per_class': recalls,
              'missing_test_classes': [CLASSES[c] for c, recall in enumerate(recalls) if recall is None],
              'limitations': 'Adjacent crops are correlated; only four source videos. '
                             'No held-out document source, so document generalization is not tested. '
                             'Requires pre-cropped objects; not an object locator.',
              'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'details': details}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output/'model.npz', weights=weights, mean=mean, scale=scale,
             classes=np.array(CLASSES))
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['scope', 'train_crops', 'test_crops', 'accuracy',
                                           'confusion', 'recall_per_class', 'missing_test_classes']}))


if __name__ == '__main__':
    main()
