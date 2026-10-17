"""Train-only diagnostic of manual object crops; not video-event accuracy."""
import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

from ml.grounded_cash import GroundedCashDetector

PROMPT = '''Identify the main object in this tightly cropped image.
Choose exactly ONE LETTER: A = banknotes, B = paper documents,
C = cash-storage object (safe, cash box, money bag), U = cannot tell.
Do not infer objects outside the crop. Answer with the letter only.'''
LABELS = {'A': 'banknotes', 'B': 'documents', 'C': 'cash_storage', 'U': None}


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-video')
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    if data['split'] != 'train':
        raise ValueError('Train diagnostic only')
    groups = collections.defaultdict(list)
    for frame in data['annotations']:
        for box in frame['boxes']:
            groups[(frame['video_id'], box['object'])].append(box['crop'])
    # One representative crop per source video and object, not every adjacent frame.
    selected = [(video, category, sorted(crops)[len(crops)//2])
                for (video, category), crops in sorted(groups.items())]
    support = []
    if args.support_video:
        support = [(video, category, name) for video, category, name in selected if video == args.support_video]
        if {category for _, category, _ in support} != {'banknotes', 'documents', 'cash_storage'}:
            raise ValueError('Support video must have all three annotated object types')
        selected = [(video, category, name) for video, category, name in selected if video != args.support_video]
    args.output.mkdir(parents=True, exist_ok=False)
    detector = GroundedCashDetector(args.model)
    support_images = [Image.open(args.annotations.parent/name).convert('RGB') for _, _, name in support]
    support_labels = [next(letter for letter, label in LABELS.items() if label == category)
                      for _, category, _ in support]
    rows = []
    for video, category, name in selected:
        start = time.monotonic()
        image = Image.open(args.annotations.parent/name).convert('RGB')
        prefix = ('The first '+str(len(support))+' images are human-labelled examples '
                  'of small objects from another video. Their labels in order are '
                  +json.dumps(support_labels)+'. Classify ONLY the LAST image.\n' if support else '')
        answer = detector.ask(prefix+PROMPT, support_images+[image], 8)
        # Metadata/paths/classes are used only for scoring, never in the model prompt.
        match = re.fullmatch(r'\s*([ABCU])\s*', answer)
        prediction = LABELS[match[1]] if match else None
        rows.append({'video_id': video, 'target': category, 'prediction': prediction,
                     'correct': prediction == category, 'width': image.width, 'height': image.height,
                     'elapsed_sec': round(time.monotonic()-start, 3)})
        (args.output/f'{len(rows):02d}-answer.txt').write_text(answer, encoding='utf-8')
    report = {'scope': 'TRAIN object-crop diagnostic, NOT event/video accuracy',
              'n_crops': len(rows), 'n_source_videos': len({r['video_id'] for r in rows}),
              'correct': sum(r['correct'] for r in rows),
              'accuracy': sum(r['correct'] for r in rows)/len(rows), 'rows': rows,
              'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              'support_video': args.support_video, 'support_labels': support_labels,
              'model': str(args.model.resolve()), 'prompt': PROMPT}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['scope', 'n_crops', 'n_source_videos', 'correct', 'accuracy', 'rows']}))


if __name__ == '__main__':
    main()
