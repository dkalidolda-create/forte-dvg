"""Replay all reviewed source files through the real local analysis pipeline.

Reference labels are read only for scoring after inference. No frames or
generated descriptions are printed or included in this compact report.
"""
import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from backend.app.pipeline.analyze import analyze_video
from ml.evaluation.train_temporal_events import confusion
from ml.rules import RULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudication', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--analyzer', default='ml.temporal_inference:analyze')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads(args.adjudication.read_text())['records']
    provenance = json.loads(args.provenance.read_text())['records']
    categories = {r['video_id']: r['categories'] for r in provenance if 'categories' in r}
    duplicates = {r['video_id']: r['duplicate_of'] for r in provenance if r.get('duplicate_of')}
    for duplicate, original in duplicates.items():
        categories[duplicate] = categories[original]
    titles = {r.title: key for key, r in RULES.items()}
    details, expected, predicted = [], [], []
    for row in records[:args.limit]:
        started = time.monotonic()
        error, answer, found, frame_count = None, None, [], 0
        try:
            answer = analyze_video(Path(row['video_path']), args.analyzer)
            for event in answer['violations']:
                title = event['type'].split(': ', 1)[1]
                found.append(titles[title])
                if not event['rule'] or not event['source_url'] or not event.get('source_section'):
                    raise ValueError('incomplete_source')
                if not 1 <= len(event['evidence_frames']) <= 3:
                    raise ValueError('invalid_evidence_count')
                for frame in event['evidence_frames']:
                    raw = base64.b64decode(frame['image_base64'], validate=True)
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or not raw.startswith(b'\xff\xd8'):
                        raise ValueError('invalid_evidence_jpeg')
                    frame_count += 1
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        target = int(row['training_verdict'] == 'violation')
        prediction = int(answer['verdict'] == 'violation') if answer and not error else 1-target
        expected.append(target)
        predicted.append(prediction)
        detail = {'video_id': row['video_id'], 'expected': row['training_verdict'],
                  'predicted': answer['verdict'] if answer and not error else None,
                  'correct_verdict': error is None and target == prediction,
                  'expected_categories': categories[row['video_id']], 'predicted_categories': found,
                  'correct_categories': error is None and set(categories[row['video_id']]) == set(found),
                  'timecodes_sec': [e['timecode_sec'] for e in answer['violations']] if answer else [],
                  'evidence_frames': frame_count, 'error': error,
                  'elapsed_sec': round(time.monotonic()-started, 3)}
        details.append(detail)
        (args.output/'progress.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
        print(json.dumps({k: detail[k] for k in ['video_id', 'correct_verdict', 'correct_categories', 'evidence_frames', 'error', 'elapsed_sec']}), flush=True)
    report = {'scope': 'TRAINING REPLAY, NOT independent accuracy', 'analyzer': args.analyzer,
              'verdict': confusion(expected, predicted),
              'coverage': sum(r['error'] is None for r in details)/len(details),
              'exact_category_accuracy': sum(r['correct_categories'] for r in details)/len(details),
              'mean_processing_sec': sum(r['elapsed_sec'] for r in details)/len(details),
              'max_processing_sec': max(r['elapsed_sec'] for r in details),
              'timing': 'Weak supervision, not independently evaluated for every event', 'details': details}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'details'}), flush=True)


if __name__ == '__main__':
    main()
