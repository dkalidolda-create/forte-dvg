"""Source-separated crop classification with existing local pretrained vision weights."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ml.vision_crops import CLASSES, FrozenVisionFeatures, RidgeCropClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-video', default='X-5')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder', choices=['qwen', 'dino'], default='qwen')
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text())
    if annotations['split'] != 'train':
        raise ValueError('Use train annotations only')
    rows = [{'video_id': row['video_id'], 'target': box['object'],
             'crop': box['crop'], 'timestamp_sec': row['timestamp_sec']}
            for row in annotations['annotations'] for box in row['boxes']]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/'source'
    source.mkdir()
    for path in [Path(__file__), Path('ml/vision_crops.py'), Path('ml/dino_crops.py')]:
        shutil.copy2(path, source/path.name)
    if args.encoder == 'dino':
        from ml.dino_crops import DinoFeatures
        extractor = DinoFeatures(args.model)
    else:
        extractor = FrozenVisionFeatures(args.model)
    vectors = []
    started = time.monotonic()
    for index, row in enumerate(rows):
        with Image.open(args.annotations.parent/row['crop']) as image:
            vectors.append(extractor.encode(image))
        if (index+1) % 10 == 0:
            print(json.dumps({'encoded': index+1, 'total': len(rows)}), flush=True)
    vectors = np.stack(vectors)
    np.savez(args.output/'features.npz', vectors=vectors)
    (args.output/'feature_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    train = np.array([r['video_id'] == args.train_video for r in rows])
    targets = np.array([CLASSES.index(r['target']) for r in rows])
    if not train.any() or not (~train).any():
        raise ValueError('Need distinct train and evaluation sources')
    classifier = RidgeCropClassifier().fit(vectors[train], targets[train])
    classifier.save(args.output/'classifier.npz')
    predictions = classifier.predict(vectors[~train])
    expected = targets[~train]
    confusion = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(expected, predictions):
        confusion[actual, predicted] += 1
    details = [{**r, 'prediction': CLASSES[int(pred)]}
               for r, pred in zip([r for r, selected in zip(rows, train) if not selected], predictions)]
    report = {'scope': 'Object-crop source-separated diagnostic inside TRAIN. NOT video-event accuracy.',
              'train_video': args.train_video, 'train_crops': int(train.sum()),
              'test_videos': sorted({r['video_id'] for r in details}), 'test_crops': len(expected),
              'accuracy': float((predictions == expected).mean()), 'classes': CLASSES,
              'confusion': confusion.tolist(), 'details': details,
              'feature_extraction_sec': round(time.monotonic()-started, 3),
              'model_load_sec': round(extractor.load_sec, 3),
              'model': str(args.model.resolve()),
              'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'regularization': .1, 'encoder': args.encoder,
              'feature': ('Normalized CLS + mean patch features' if args.encoder == 'dino' else
                          'Normalized pooled final + deepstack vision features'),
              'limitations': 'Only four source videos; correlated adjacent crops; no held-out '
              'document source. Objects are manually located; not an automatic object detector.'}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['scope', 'train_crops', 'test_crops',
                                           'accuracy', 'confusion', 'feature_extraction_sec']}))


if __name__ == '__main__':
    main()
