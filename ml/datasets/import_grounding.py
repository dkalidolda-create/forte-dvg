"""Validate manual train annotations and create local object crops, without relabelling videos."""
import argparse
import json
import math
from pathlib import Path


def validate_annotations(payload, review):
    if payload.get('schema_version') != 1 or payload.get('split') != 'train':
        raise ValueError('Expected schema 1, train only')
    if payload.get('manifest_sha256') != review['manifest_hash']:
        raise ValueError('Annotations belong to another dataset version')
    known = {(v['video_id'], f['timestamp_sec']): f for v in review['videos'] for f in v['frames']}
    rows = payload.get('annotations')
    if not isinstance(rows, list) or not rows:
        raise ValueError('No manual annotations')
    seen = set()
    clean = []
    for row in rows:
        key = (row.get('video_id'), row.get('timestamp_sec'))
        if key not in known or key in seen:
            raise ValueError('Unknown or duplicate training frame')
        seen.add(key)
        if row.get('operation_state') not in ['unreviewed', 'active', 'ended', 'unclear']:
            raise ValueError('Invalid operation state')
        note = row.get('note', '')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('Invalid note')
        boxes = row.get('boxes')
        if not isinstance(boxes, list) or len(boxes) > 100:
            raise ValueError('Invalid object list')
        for box in boxes:
            if box.get('object') not in ['banknotes', 'documents', 'cash_storage']:
                raise ValueError('Invalid object category')
            coords = [box.get(k) for k in ['x', 'y', 'width', 'height']]
            if any(type(v) not in [int, float] or not math.isfinite(v) for v in coords):
                raise ValueError('Non-finite or non-numeric box coordinates')
            x, y, width, height = coords
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > 1.000001 or y+height > 1.000001:
                raise ValueError('Box outside frame')
        frame = known[key]
        clean.append({'video_id': key[0], 'timestamp_sec': key[1],
                      'image': frame['image'], 'operation_state': row['operation_state'],
                      'boxes': boxes, 'note': note})
    return clean


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads((args.review_root/'frames.json').read_text())
    payload = json.loads(args.annotations.read_text())
    clean = validate_annotations(payload, review)
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    for index, row in enumerate(clean):
        image_path = (args.review_root/row['image']).resolve()
        if not image_path.is_relative_to(args.review_root.resolve()):
            raise ValueError('Invalid image path')
        image = Image.open(image_path)
        for number, box in enumerate(row['boxes']):
            x, y, width, height = [box[k] for k in ['x', 'y', 'width', 'height']]
            crop = image.crop((int(x*image.width), int(y*image.height),
                               max(int(x*image.width)+1, math.ceil((x+width)*image.width)),
                               max(int(y*image.height)+1, math.ceil((y+height)*image.height))))
            name = f'{index:03d}-{number:02d}-{box["object"]}.jpg'
            crop.save(args.output/name, quality=95)
            box['crop'] = name
            count += 1
    (args.output/'annotations.json').write_text(json.dumps({
        'schema_version': 1, 'split': 'train', 'manifest_sha256': review['manifest_hash'],
        'annotations': clean,
        'note': 'Manual object crops and operation states. Original video verdicts unchanged. '
                'Unmarked objects are not negative annotations; not a complete detector-training dataset.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'annotated_frames': len(clean), 'object_crops': count, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
