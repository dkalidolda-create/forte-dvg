"""Build offline visual SFT examples from all human-reviewed training videos.

Do not train legal claims: output category codes only. Images and labels stay local.
Final all-data training and development-separated training are distinct folders.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from ml.event_schema import prompt
from ml.grounded_cash import choose_indexes
from ml.local_cash import sample_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.adjudication.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    image_root = args.output/'images'
    image_root.mkdir()
    split_root, all_root = args.output/'development', args.output/'all-data'
    split_root.mkdir()
    all_root.mkdir()
    seen, split_rows, provenance = {}, {'train': [], 'validation': []}, []
    for record in data['records']:
        group = record['exact_file_group']
        if group in seen:
            provenance.append({'video_id': record['video_id'], 'duplicate_of': seen[group], 'included_via_duplicate_group': True})
            continue
        seen[group] = record['video_id']
        path = Path(record['video_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Changed video')
        frames, times, duration = sample_video(path, 1., 448, time.monotonic()+120)
        indexes = choose_indexes(len(frames), 6)
        chosen_times = [times[i] for i in indexes]
        images = []
        for order, index in enumerate(indexes):
            destination = image_root/f'{len(seen):03d}-{order}.jpg'
            frames[index].save(destination, 'JPEG', quality=92)
            images.append(str(destination.resolve()))
        categories = list(record['categories'])
        if record['video_id'] == 'X-38' and record.get('model_feedback') == 'correct':
            # The reviewed live model exposed only its cash event category.
            categories = ['cash_left_after_service']
        target = record['training_verdict']
        if target == 'violation' and not categories:
            raise ValueError('Positive record without a category: '+record['video_id'])
        if target == 'no_violation' and categories:
            raise ValueError('Negative record has positive categories')
        annotated = record.get('organizer_time_sec')
        # One organizer timestamp does not localize every event in a combination.
        # This is explicitly weak localization supervision, never ground-truth accuracy.
        frame = (min(range(len(chosen_times)), key=lambda i: abs(chosen_times[i]-float(annotated)))
                 if annotated else len(chosen_times)-1)
        events = [{'category': c, 'frame': frame} for c in categories]
        answer = {'verdict': target, 'events': events}
        example = {'messages': [{'role': 'user', 'content': prompt(chosen_times)},
                                {'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)}],
                   'images': images}
        split_rows[record['split']].append(example)
        provenance.append({'video_id': record['video_id'], 'sha256': record['sha256'],
                           'split': record['split'], 'target': target, 'categories': categories,
                           'category_provenance': 'correct_live_cash_model_feedback' if record['video_id'] == 'X-38' else record['category_provenance'],
                           'frame_timestamps': chosen_times, 'duration_sec': duration,
                           'timestamp_supervision': 'weak: organizer timestamp or last sampled frame; combinations not independently timed',
                           'user_override': bool(record.get('training_override'))})
    for split, rows in split_rows.items():
        (split_root/(split+'.jsonl')).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n')
    all_rows = split_rows['train']+split_rows['validation']
    (all_root/'train.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in all_rows)+'\n')
    metadata = {'reviewed_records': len(data['records']), 'unique_videos': len(seen),
                'development_counts': {k: len(v) for k,v in split_rows.items()},
                'all_data_training_examples': len(all_rows), 'all_categories': sorted({c for p in provenance for c in p.get('categories', [])}),
                'records': provenance, 'legal_status': 'Provisional visual events, not adjudicated legal violations',
                'evaluation_policy': 'All-data training consumes development validation; subsequent same-video scores are resubstitution only',
                'adjudication_sha256': hashlib.sha256(args.adjudication.read_bytes()).hexdigest()}
    (args.output/'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k:metadata[k] for k in ['reviewed_records','unique_videos','development_counts','all_data_training_examples','all_categories']}))


if __name__ == '__main__':
    main()
