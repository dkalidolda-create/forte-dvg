"""Train local binary and partially labelled event heads; report development metrics.

Never called by the website. No labels, filenames or hashes are runtime inputs.
Normative eligibility is deliberately independent of an organizer's binary label.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.dino_crops import DinoFeatures
from ml.rules import RULES
from ml.temporal_features import binary_scores, extract, fit_binary


def confusion(expected, predicted):
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    tp = int(((expected == 1) & (predicted == 1)).sum())
    tn = int(((expected == 0) & (predicted == 0)).sum())
    fp = int(((expected == 0) & (predicted == 1)).sum())
    fn = int(((expected == 1) & (predicted == 0)).sum())
    return {'n': len(expected), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp+tn)/len(expected) if len(expected) else None,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudication', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.adjudication.read_text())
    records, seen = [], set()
    for row in data['records']:
        if (row['label_conflict'] and not row.get('training_override')) or row['exact_file_group'] in seen:
            continue
        seen.add(row['exact_file_group'])
        records.append(row)
    train = np.array([r['split'] == 'train' for r in records])
    if not train.any() or not (~train).any():
        raise ValueError('Need separate video groups')
    encoder = DinoFeatures(args.encoder)
    cache = args.output/'features'
    cache.mkdir()
    vectors, extraction = [], []
    for i, row in enumerate(records):
        path = Path(row['video_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Video changed after split assignment')
        vector, metadata = extract(path, encoder)
        vectors.append(vector)
        extraction.append({'video_id': row['video_id'], **metadata})
        np.save(cache/(row['video_id']+'.npy'), vector, allow_pickle=False)
        print(json.dumps({'encoded': i+1, 'total': len(records), 'video_id': row['video_id'],
                          'seconds': metadata['elapsed_sec']}), flush=True)
    vectors = np.asarray(vectors)
    targets = np.array([int(r['training_verdict'] == 'violation') for r in records])
    support, dual = fit_binary(vectors[train], targets[train])
    np.savez(args.output/'binary-head.npz', support=support, dual=dual)
    scores = binary_scores(vectors, support, dual)
    validation = confusion(targets[~train], scores[~train] >= 0)
    predictions = [{'video_id': r['video_id'], 'split': r['split'], 'target': int(t),
                    'prediction': int(score >= 0), 'raw_score': round(float(score), 6)}
                   for r, t, score in zip(records, targets, scores)]
    # Only whole-video negatives are negative examples for each event. Unlisted
    # categories in a positive video remain unknown, never silently negative.
    heads = []
    for category, rule in RULES.items():
        known = np.array([category in r['categories'] or r['training_verdict'] == 'no_violation'
                          for r in records])
        target = np.array([int(category in r['categories']) for r in records])
        selected = train & known
        positives = int(target[selected].sum())
        if positives == 0:
            heads.append({'category': category, 'status': 'no_training_positive'})
            continue
        s, d = fit_binary(vectors[selected], target[selected])
        np.savez(args.output/(category+'.npz'), support=s, dual=d)
        mask = (~train) & known
        pred = binary_scores(vectors[mask], s, d) >= 0
        heads.append({'category': category, 'status': 'experimental_only', 'training_positive_videos': positives,
                      'training_negative_videos': int((1-target[selected]).sum()),
                      'normative_relation': rule.relation, 'validation': confusion(target[mask], pred),
                      'limitation': 'Rare labels and scene features; no independently verified event localization.'})
    report = {'scope': 'Reused development split, NOT independent test accuracy',
              'training_videos': int(train.sum()), 'validation_videos': int((~train).sum()),
              'excluded_conflicts': data['summary']['label_conflicts'],
              'features': 'Frozen DINOv2 CLS+mean patches, 12 chronological frames, full image + four quadrants; 3 phases + signed change + motion magnitude',
              'encoder_weights_updated': False, 'trained_component': 'Balanced kernel-ridge binary and partial-label event heads',
              'threshold': 0, 'alpha': 1.0, 'validation': validation, 'heads': heads,
              'predictions': predictions, 'extraction': extraction,
              'deployment': 'NOT enabled: scene scores do not establish visible evidence, type or legal applicability',
              'adjudication_sha256': hashlib.sha256(args.adjudication.read_bytes()).hexdigest(),
              'encoder_sha256': hashlib.sha256((args.encoder/'model.onnx').read_bytes()).hexdigest()}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ['scope', 'training_videos', 'validation_videos', 'validation', 'deployment']}), flush=True)


if __name__ == '__main__':
    main()
