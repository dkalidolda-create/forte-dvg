"""Build category-specific truth separately from inference, using reviewed cases."""
import argparse
import csv
from pathlib import Path

# Explicitly confirmed by the team. Other positive categories are NOT negatives.
# X-15 withdrawn from this category after frame review: no banknotes visible.
# Its original organizer verdict is preserved in the general manifest.
CONFIRMED_POSITIVES = {'X-4', 'X-26', 'X-44'}
DISPUTED = {'X-12'}
UNCONFIRMED_CATEGORY = {'X-15'}


def prepare(manifest, output):
    with manifest.open(encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    selected, seen = [], set()
    for row in rows:
        key = row['video_id']
        if key in DISPUTED or key in UNCONFIRMED_CATEGORY:
            continue
        if key not in CONFIRMED_POSITIVES and row['target'] != '0':
            continue
        if row['sha256'] in seen:
            continue
        seen.add(row['sha256'])
        selected.append({'video_id': key, 'video_path': row['video_path'],
                         'split': row['split'], 'sha256': row['sha256'],
                         'category': 'cash_left_after_operation',
                         'target': int(key in CONFIRMED_POSITIVES),
                         'event_sec': row['event_sec'] if key in CONFIRMED_POSITIVES else '',
                         'label_source': 'team_confirmed' if key in CONFIRMED_POSITIVES else 'organizer_no_violation'})
    if output.exists():
        raise ValueError('Do not overwrite frozen ground truth')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print({'rows': len(selected), 'positives': sum(r['target'] for r in selected),
           'validation': sum(r['split'] == 'validation' for r in selected)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
