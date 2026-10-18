"""Diagnose binary decoding bias with training-only threshold selection.

Validation is never used to choose the threshold. Scores and predictions stay
local. The first experiment must use the development-only adapter, not the
all-data checkpoint that has already seen validation.
"""
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

from ml.evaluation.train_temporal_events import confusion
from ml.multi_event import EventDetector


def choose_threshold(scores, targets):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if not np.isfinite(scores).all() or set(targets.tolist()) != {0, 1}:
        raise ValueError('Finite scores and both training classes required')
    values = sorted(set(scores))
    thresholds = [values[0]-1]+[(a+b)/2 for a,b in zip(values, values[1:])]+[values[-1]+1]
    def objective(threshold):
        predictions = scores >= threshold
        tp = ((predictions == 1) & (targets == 1)).sum()
        tn = ((predictions == 0) & (targets == 0)).sum()
        balanced = .5*(tp/(targets == 1).sum()+tn/(targets == 0).sum())
        return balanced, tn, threshold
    return max(thresholds, key=objective)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    detector = EventDetector(args.model, args.adapter)
    records, by_split = [], {}
    for split in ['train', 'validation']:
        path = args.dataset/(split+'.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_split[split] = rows
        for i,row in enumerate(rows):
            images = []
            try:
                for image_path in row['images']:
                    with Image.open(image_path) as image:
                        images.append(image.convert('RGB').copy())
                question = row['messages'][0]['content']
                times = json.loads(question.rsplit('Chronological frame timestamps in seconds: ',1)[1])
                target = int(json.loads(row['messages'][-1]['content'])['verdict'] == 'violation')
                margin = detector.decision_margin(images, times)
                records.append({'split':split,'sample':i,'target':target,'margin':margin})
                print(json.dumps({'scored':len(records),'split':split}),flush=True)
            finally:
                for image in images:
                    image.close()
    train = [r for r in records if r['split']=='train']
    threshold = choose_threshold([r['margin'] for r in train],[r['target'] for r in train])
    metrics = {split:confusion([r['target'] for r in records if r['split']==split],
                              [r['margin']>=threshold for r in records if r['split']==split]) for split in by_split}
    report = {'scope':'Training-only threshold; reused development validation, not independent test',
              'threshold':threshold,'metrics':metrics,'records':records,
              'adapter_sha256':hashlib.sha256((args.adapter/'adapters.safetensors').read_bytes()).hexdigest(),
              'deployment':'Not enabled automatically; type and evidence validation still required'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['scope','threshold','metrics']}),flush=True)


if __name__ == '__main__':
    main()
