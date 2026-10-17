"""Score predictions separately; failed/uncertain videos are never silently dropped."""
import argparse
import csv
import json
from pathlib import Path


def score(truth, predictions):
    expected = {r['video_id']: r for r in truth}
    if len(expected) != len(truth) or not truth:
        raise ValueError('Truth must contain unique IDs and at least one video')
    found = {}
    for prediction in predictions:
        key = prediction['video_id']
        if key not in expected or key in found:
            raise ValueError('Unknown or duplicate prediction ID')
        value = prediction.get('prediction')
        if value is not None and (type(value) is not int or value not in [0, 1]):
            raise ValueError('Prediction must be integer 0, 1 or null')
        found[key] = prediction
    tp = tn = fp = fn = unresolved_positive = unresolved_negative = 0
    details, errors = [], []
    for key, row in expected.items():
        actual = int(row['target'])
        prediction = found.get(key, {'status': 'missing', 'prediction': None})
        value = prediction.get('prediction') if prediction['status'] == 'done' else None
        if value is None:
            if actual:
                unresolved_positive += 1
            else:
                unresolved_negative += 1
        elif actual and value:
            tp += 1
        elif not actual and not value:
            tn += 1
        elif not actual and value:
            fp += 1
        else:
            fn += 1
        error = None
        if actual and value == 1 and row.get('event_sec') not in ['', None]:
            times = prediction.get('event_times_sec', [])
            if times:
                error = min(abs(float(t)-float(row['event_sec'])) for t in times)
                errors.append(error)
        details.append({'video_id': key, 'target': actual, 'prediction': value,
                        'status': prediction['status'], 'correct': value is not None and value == actual,
                        'time_error_sec': error, 'elapsed_sec': prediction.get('elapsed_sec')})
    n = len(expected)
    resolved = tp + tn + fp + fn
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn+unresolved_positive) if tp+fn+unresolved_positive else None
    specificity = tn/(tn+fp+unresolved_negative) if tn+fp+unresolved_negative else None
    balanced = (recall+specificity)/2 if recall is not None and specificity is not None else None
    f1 = (2*precision*recall/(precision+recall)
          if precision is not None and recall is not None and precision+recall else
          (0.0 if precision is not None and recall is not None else None))
    report = {'category': 'cash_left_after_operation', 'n_expected': n,
              'n_positive': sum(int(r['target']) for r in truth),
              'n_negative': sum(1-int(r['target']) for r in truth),
              'n_resolved': resolved, 'coverage': resolved/n if n else None,
              'accuracy_all_videos': (tp+tn)/n if n else None,
              'accuracy_resolved_only': (tp+tn)/resolved if resolved else None,
              'precision': precision, 'recall_conservative': recall, 'f1_conservative': f1,
              'specificity_conservative': specificity, 'balanced_accuracy_conservative': balanced,
              'tp': tp, 'tn': tn, 'fp': fp, 'fn_resolved': fn,
              'unresolved_positive': unresolved_positive, 'unresolved_negative': unresolved_negative,
              'mean_time_error_sec_matched_only': sum(errors)/len(errors) if errors else None,
              'localized_true_positives': len(errors),
              'always_negative_accuracy': sum(1-int(r['target']) for r in truth)/n,
              'always_positive_accuracy': sum(int(r['target']) for r in truth)/n,
              'n_failed': sum(r['status'] == 'failed' for r in details),
              'n_missing': sum(r['status'] == 'missing' for r in details),
              'max_elapsed_sec': max((r['elapsed_sec'] for r in details if r['elapsed_sec'] is not None), default=None),
              'note': 'Category-specific exploratory evaluation. Unresolved cases count against all-video accuracy. Not the official jury score.'}
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth = [r for r in csv.DictReader(args.manifest.open(encoding='utf-8-sig'))
             if r['split'] == args.split and (not args.ids or r['video_id'] in args.ids)]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    report, details = score(truth, predictions)
    report['split'] = args.split
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'per_video.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
