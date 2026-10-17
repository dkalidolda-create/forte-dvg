"""Point/patch supervision from manually located TRAIN objects; unmarked patches ignored."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ml.dino_crops import DinoFeatures
from ml.vision_crops import RidgeCropClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    if data['split'] != 'train':
        raise ValueError('Train annotations required')
    corrections = json.loads(args.corrections.read_text())
    bags = {r['video_id'] for r in corrections['records'] if r['category'] == 'collection_bag_left_unattended'}
    classes = ['banknotes', 'documents', 'cash_storage', 'collection_bag']
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = DinoFeatures(args.model)
    features, labels, sources = [], [], []
    started = time.monotonic()
    for index, row in enumerate(data['annotations']):
        if not row['boxes']:
            continue
        with Image.open(args.review_root/row['image']) as image:
            for box in row['boxes']:
                category = 'collection_bag' if row['video_id'] in bags and box['object'] == 'cash_storage' else box['object']
                cx = (box['x']+box['width']/2)*image.width
                cy = (box['y']+box['height']/2)*image.height
                left = max(0, min(image.width-224, round(cx-112)))
                top = max(0, min(image.height-224, round(cy-112)))
                dense = encoder.dense_tile(image.crop((left, top, left+224, top+224)))
                ys, xs = np.meshgrid((top+(np.arange(16)+.5)*14)/image.height,
                                     (left+(np.arange(16)+.5)*14)/image.width, indexing='ij')
                inside = ((xs >= box['x']) & (xs <= box['x']+box['width'])
                          & (ys >= box['y']) & (ys <= box['y']+box['height'])).ravel()
                distance = ((xs-box['x']-box['width']/2)**2+(ys-box['y']-box['height']/2)**2).ravel()
                positions = np.flatnonzero(inside)
                if not len(positions):
                    continue  # Never substitute a patch outside the manual rectangle.
                positions = positions[np.argsort(distance[positions])[:8]]
                features.extend(dense[positions])
                labels.extend([classes.index(category)]*len(positions))
                sources.extend([row['video_id']]*len(positions))
        if (index+1) % 10 == 0:
            print(json.dumps({'processed_frames': index+1, 'total': len(data['annotations'])}), flush=True)
    vectors, targets = np.stack(features), np.array(labels)
    source_array = np.array(sources)
    train = source_array != 'X-27'
    classifier = RidgeCropClassifier(classes).fit(vectors[train], targets[train])
    predictions = classifier.predict(vectors[~train])
    confusion = np.zeros((4, 4), int)
    for a, b in zip(targets[~train], predictions):
        confusion[a, b] += 1
    final = RidgeCropClassifier(classes).fit(vectors, targets)
    final.save(args.output/'classifier.npz')
    np.savez(args.output/'support.npz', vectors=vectors, targets=targets)
    metadata = {'scope': 'Dense object patch component; not video/event accuracy.',
                'classes': classes, 'patches': len(targets), 'annotated_frames': len(data['annotations']),
                'diagnostic_source': 'X-27', 'diagnostic_patches': int((~train).sum()),
                'diagnostic_accuracy': float((targets[~train] == predictions).mean()),
                'diagnostic_confusion': confusion.tolist(),
                'diagnostic_note': 'Dependent patches from one train video; no independent test.',
                'final_training_sources': sorted(set(sources)), 'tile_size': 224, 'patch_size': 14,
                'elapsed_sec': round(time.monotonic()-started, 3),
                'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                'corrections_sha256': hashlib.sha256(args.corrections.read_bytes()).hexdigest()}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    folder = args.output/'source'
    folder.mkdir()
    for path in [Path(__file__), Path('ml/dino_crops.py'), Path('ml/vision_crops.py')]:
        shutil.copy2(path, folder/path.name)
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
