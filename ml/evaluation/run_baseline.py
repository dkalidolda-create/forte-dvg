"""Run offline inference; query labels/timestamps are withheld from the detector."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--example-ids', nargs='*', default=[])
    parser.add_argument('--detector', choices=['legacy', 'grounded', 'panels', 'roi', 'fast', 'dense'], default='legacy')
    parser.add_argument('--manual-annotations', type=Path)
    parser.add_argument('--crop-model', type=Path)
    parser.add_argument('--dino-model', type=Path)
    parser.add_argument('--max-side', type=int, default=640)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory; previous runs are immutable')
    rows = list(csv.DictReader(args.manifest.open(encoding='utf-8-sig')))
    # Deliberately strip labels before constructing the inference workload.
    tasks = [(r['video_id'], Path(r['video_path'])) for r in rows
             if r['split'] == args.split and (not args.ids or r['video_id'] in args.ids)]
    if not tasks:
        raise ValueError('No videos selected')
    if args.ids and set(args.ids) != {key for key, _ in tasks}:
        raise ValueError('Some IDs are missing or belong to another split')
    examples = [r for r in rows if r['video_id'] in args.example_ids]
    if {r['video_id'] for r in examples} != set(args.example_ids):
        raise ValueError('Missing demonstration ID')
    task_ids = {key for key, _ in tasks}
    task_hashes = {r['sha256'] for r in rows if r['video_id'] in task_ids}
    if any(r['split'] != 'train' or r['sha256'] in task_hashes for r in examples):
        raise ValueError('Demonstrations must be train-only and disjoint from evaluated videos')
    args.output.mkdir(parents=True)
    extra = {}
    if args.detector in ['roi', 'fast', 'dense']:
        if not args.manual_annotations:
            raise ValueError('ROI detector requires train manual annotations')
        if args.detector in ['fast', 'dense']:
            if not args.crop_model or not args.dino_model:
                raise ValueError('Fast detector requires crop and DINO model directories')
            if args.detector == 'dense':
                from ml.dense_cash import DenseCashDetector as Detector
                code_path = Path('ml/dense_cash.py')
            else:
                from ml.fast_cash import FastCashDetector as Detector
                code_path = Path('ml/fast_cash.py')
            extra.update(crop_model=args.crop_model, dino_model=args.dino_model)
            metadata = json.loads((args.crop_model/'metadata.json').read_text())
            if set(metadata['final_training_sources']) & {r['video_id'] for r in rows if r['split'] == 'validation'}:
                raise ValueError('Crop model has validation source leakage')
        else:
            from ml.roi_cash import RoiCashDetector as Detector
            code_path = Path('ml/roi_cash.py')
        calibration = json.loads(args.manual_annotations.read_text())
        calibration_ids = {r['video_id'] for r in calibration['annotations']
                           if any(b['object'] == 'banknotes' for b in r['boxes'])}
        train_ids = {r['video_id'] for r in rows if r['split'] == 'train'}
        if calibration.get('split') != 'train' or not calibration_ids <= train_ids:
            raise ValueError('ROI banknote calibration contains non-train videos')
        extra['annotations'] = args.manual_annotations
    elif args.detector == 'panels':
        from ml.example_cash import ExampleCashDetector as Detector
        code_path = Path('ml/example_cash.py')
    elif args.detector == 'grounded':
        from ml.grounded_cash import GroundedCashDetector as Detector
        code_path = Path('ml/grounded_cash.py')
    else:
        from ml.local_cash import CashDetector as Detector
        code_path = Path('ml/local_cash.py')
    if not 320 <= args.max_side <= 1280:
        raise ValueError('Use max-side between 320 and 1280')
    detector = Detector(args.model, max_side=args.max_side,
                        examples=[(Path(r['video_path']), int(r['target'])) for r in examples], **extra)
    source_dir = args.output / 'source'
    source_dir.mkdir()
    for source in {code_path, Path('ml/local_cash.py'), Path('ml/grounded_cash.py'),
                   Path('ml/fast_cash.py'), Path('ml/vision_crops.py'), Path('ml/dino_crops.py'), Path(__file__)}:
        shutil.copy2(source, source_dir / source.name)
    provenance = {'config': detector.config, 'model_load_sec': detector.load_sec,
                  'split': args.split, 'video_ids': [key for key, _ in tasks],
                  'example_ids': args.example_ids,
                  'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  'detector_code_sha256': hashlib.sha256(code_path.read_bytes()).hexdigest(),
                  'shared_code_sha256': hashlib.sha256(Path('ml/local_cash.py').read_bytes()).hexdigest()}
    (args.output / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as file:
        for key, path in tasks:
            start = time.monotonic()
            try:
                result = {'video_id': key, 'status': 'done',
                          **detector.analyze(path, args.output / key)}
            except Exception as exc:
                result = {'video_id': key, 'status': 'failed', 'prediction': None,
                          'elapsed_sec': round(time.monotonic()-start, 3),
                          'error_type': type(exc).__name__}
                # Raw validation failures can contain model descriptions: save locally only.
                (args.output / f'{key}-error.txt').write_text(str(exc), encoding='utf-8')
            file.write(json.dumps(result) + '\n')
            file.flush()
            # Only identifiers and process status; do not print video observations.
            print(json.dumps({'video_id': key, 'status': result['status'],
                              'elapsed_sec': result['elapsed_sec']}), flush=True)


if __name__ == '__main__':
    main()
