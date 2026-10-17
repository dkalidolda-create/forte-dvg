"""Fit corrected object classes from cached TRAIN features, preserving a source-held diagnostic."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from ml.vision_crops import RidgeCropClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.features/'feature_rows.json').read_text())
    vectors = np.load(args.features/'features.npz')['vectors']
    annotations = json.loads(args.annotations.read_text())
    if annotations['split'] != 'train' or len(vectors) != len(rows):
        raise ValueError('Invalid training features')
    corrections = json.loads(args.corrections.read_text())
    bag_sources = {r['video_id'] for r in corrections['records']
                   if r['category'] == 'collection_bag_left_unattended'}
    classes = ['banknotes', 'documents', 'cash_storage', 'collection_bag']
    for row in rows:
        if row['video_id'] in bag_sources and row['target'] == 'cash_storage':
            row['original_object_class'] = row['target']
            row['target'] = 'collection_bag'
    targets = np.array([classes.index(r['target']) for r in rows])
    train = np.array([r['video_id'] != 'X-27' for r in rows])
    diagnostic = RidgeCropClassifier(classes).fit(vectors[train], targets[train])
    prediction = diagnostic.predict(vectors[~train])
    expected = targets[~train]
    confusion = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(expected, prediction):
        confusion[a, b] += 1
    # After the internal diagnostic, final component fits all TRAIN object annotations.
    final = RidgeCropClassifier(classes).fit(vectors, targets)
    args.output.mkdir(parents=True, exist_ok=False)
    final.save(args.output/'classifier.npz')
    np.savez(args.output/'support.npz', vectors=vectors, targets=targets)
    (args.output/'support_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    report = {'scope': 'TRAIN object component, not event/video accuracy',
              'classes': classes, 'diagnostic_source': 'X-27',
              'diagnostic_crops': len(expected), 'diagnostic_accuracy': float((prediction == expected).mean()),
              'diagnostic_confusion': confusion.tolist(),
              'cash_vs_other_accuracy': float(((prediction == 0) == (expected == 0)).mean()),
              'diagnostic_note': 'Development choice; six correlated crops from one train video. '
              'Not an independent final test; documents and collection bags absent from diagnostic.',
              'final_training_crops': len(rows), 'final_training_sources': sorted({r['video_id'] for r in rows}),
              'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'corrections_sha256': hashlib.sha256(args.corrections.read_bytes()).hexdigest(),
              'features_metadata': str(args.features/'metrics.json')}
    (args.output/'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    source = args.output/'source'
    source.mkdir()
    for path in [Path(__file__), Path('ml/vision_crops.py')]:
        shutil.copy2(path, source/path.name)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
