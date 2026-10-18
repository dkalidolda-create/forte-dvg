"""Fit visual heads and record honest development and training-replay scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.evaluation.train_temporal_events import confusion
from ml.temporal_events import TemporalEventHead


def evaluate(model, vectors, targets, category_targets, categories):
    answers = [model.predict(v) for v in vectors]
    predicted = [int(a['verdict'] == 'violation') for a in answers]
    exact = sum({e['category'] for e in answer['events']} ==
                {categories[j] for j in np.flatnonzero(row)}
                for answer, row in zip(answers, category_targets))
    return {'verdict': confusion(targets, predicted), 'exact_category_accuracy': exact/len(answers),
            'coverage': 1.0, 'answers': answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(args.provenance.read_text())
    records = [r for r in provenance['records'] if not r.get('duplicate_of')]
    categories = sorted(provenance['all_categories'])
    vectors = np.asarray([np.load(args.features/(r['video_id']+'.npy'), allow_pickle=False) for r in records])
    binary = np.asarray([int(r['target'] == 'violation') for r in records])
    targets = np.asarray([[int(c in r['categories']) for c in categories] for r in records])
    timing = np.asarray([[max(r['frame_timestamps'])/r['duration_sec'] if c in r['categories'] else 0
                          for c in categories] for r in records])
    # Use the actual weak timestamp selected by the SFT preparer, not a new
    # invented annotation. The same weak time applies to combination categories.
    rows = []
    root = args.provenance.parent/'all-data'/'train.jsonl'
    for line in root.read_text().splitlines():
        row = json.loads(line)
        rows.append(json.loads(row['messages'][-1]['content']))
    # All-data JSONL is train rows followed by validation rows.
    ordered = [i for split in ['train', 'validation'] for i,r in enumerate(records) if r['split'] == split]
    for index, row in zip(ordered, rows):
        for event in row['events']:
            column = categories.index(event['category'])
            timing[index, column] = records[index]['frame_timestamps'][event['frame']]/records[index]['duration_sec']
    train = np.asarray([r['split'] == 'train' for r in records])
    development = TemporalEventHead.fit(vectors[train], binary[train], targets[train], timing[train], categories)
    development.save(args.output/'development-head.npz')
    validation = evaluate(development, vectors[~train], binary[~train], targets[~train], categories)
    final = TemporalEventHead.fit(vectors, binary, targets, timing, categories)
    final.save(args.output/'model.npz')
    replay = evaluate(final, vectors, binary, targets, categories)
    report = {'method': 'Frozen DINOv2 chronology + supervised RBF kernel ridge event heads',
              'reviewed_records': provenance['reviewed_records'], 'unique_training_videos': len(records),
              'alpha': final.alpha, 'gamma': final.gamma, 'categories': categories,
              'development_validation': validation, 'all_data_training_replay': replay,
              'timing': 'Weak organizer times / inferred endpoints; not independently validated per category',
              'generalization': 'Small same-camera dataset. Training replay is NOT independent accuracy.',
              'provenance_sha256': hashlib.sha256(args.provenance.read_bytes()).hexdigest(),
              'records': [{'video_id': r['video_id'], 'split': r['split']} for r in records]}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'development': {k:v for k,v in validation.items() if k != 'answers'},
                      'training_replay': {k:v for k,v in replay.items() if k != 'answers'}}), flush=True)


if __name__ == '__main__':
    main()
