"""Evaluate the local visual adapter; never print frames or generated descriptions."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from ml.evaluation.train_temporal_events import confusion
from ml.event_schema import parse
from ml.multi_event import EventDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', required=True, choices=['development_validation', 'training_resubstitution'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    detector = EventDetector(args.model, args.adapter)
    expected, predicted, details = [], [], []
    category_tp = category_fp = category_fn = exact_types = invalid = 0
    for index, row in enumerate(rows):
        images = []
        try:
            for path in row['images']:
                with Image.open(path) as image:
                    images.append(image.convert('RGB').copy())
            question = row['messages'][0]['content']
            times = json.loads(question.rsplit('Chronological frame timestamps in seconds: ', 1)[1])
            truth = parse(row['messages'][-1]['content'], len(images))
            started = time.monotonic()
            try:
                answer = detector.classify(images, times)
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                answer = None
                error = type(exc).__name__
                invalid += 1
            target = int(truth['verdict'] == 'violation')
            # Unprocessed results always count as wrong; retain separately in coverage.
            prediction = int(answer['verdict'] == 'violation') if answer else 1-target
            expected.append(target)
            predicted.append(prediction)
            true_types = {e['category'] for e in truth['events']}
            found_types = {e['category'] for e in answer['events']} if answer else set()
            category_tp += len(true_types & found_types)
            category_fp += len(found_types-true_types)
            category_fn += len(true_types-found_types)
            exact_types += int(answer is not None and true_types == found_types)
            detail = {'sample': index, 'expected': truth, 'predicted': answer, 'error': error,
                      'elapsed_sec': round(time.monotonic()-started, 3), 'timestamps': times}
            details.append(detail)
            (args.output/f'sample-{index:03d}.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2))
            print(json.dumps({'sample': index+1, 'total': len(rows), 'correct_verdict': bool(answer and prediction == target),
                              'correct_types': bool(answer and true_types == found_types), 'error': error,
                              'elapsed_sec': detail['elapsed_sec']}), flush=True)
        finally:
            for image in images:
                image.close()
    report = {'scope': args.scope, 'not_independent_test': True, 'verdict': confusion(expected, predicted),
              'coverage': (len(rows)-invalid)/len(rows), 'invalid_outputs': invalid,
              'exact_category_accuracy': exact_types/len(rows),
              'category_counts': {'tp': category_tp, 'fp': category_fp, 'fn': category_fn},
              'category_precision': category_tp/(category_tp+category_fp) if category_tp+category_fp else None,
              'category_recall': category_tp/(category_tp+category_fn) if category_tp+category_fn else None,
              'localization': 'weak organizer timestamps / inferred endpoint, not independently verified for every category',
              'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'adapter_sha256': hashlib.sha256((args.adapter/'adapters.safetensors').read_bytes()).hexdigest() if args.adapter else None,
              'model_load_sec': detector.load_sec, 'details': details}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:report[k] for k in ['scope','verdict','coverage','exact_category_accuracy','category_precision','category_recall']}), flush=True)


if __name__ == '__main__':
    main()
