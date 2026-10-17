"""Offline train-only diagnosis; video-derived reasoning is saved locally only."""
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from ml.local_cash import (
    PROMPT,
    CashDetector,
    WindowDecision,
    parse_decision,
    sample_video,
)

GROUNDED_PROMPT = '''Examine the images as one chronological video sequence.
Decide whether the following specific event is visually supported: banknotes are
left on the cashier's work surface AFTER a customer interaction has visibly ended.
First describe briefly what visibly changes across the sequence, then distinguish
banknotes from paper documents and closed cash boxes. Identify visible evidence
that the interaction ended; a pause in counting is not sufficient.
Do not assume these images contain this event. Active counting, ongoing service,
or cash put inside a cash box are negative examples. If the small objects or
completion cannot be resolved, explicitly say the evidence is insufficient.
Do not identify people or infer legal guilt, ownership or missing paperwork.
Conclude on the LAST LINE with exactly one of:
FINAL: A (event supported), FINAL: B (event not observed), FINAL: U (unclear).
Frame timestamps in seconds: '''


def parse_final(text):
    matches = re.findall(r'^\s*FINAL:\s*([ABU])\b', text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous final decision')
    return {'A': 1, 'B': 0, 'U': None}[matches[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['X-5', 'X-15', 'X-26', 'X-27'])
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding='utf-8-sig')))
    selected = [r for r in rows if r['video_id'] in args.ids]
    if {r['video_id'] for r in selected} != set(args.ids) or any(r['split'] != 'train' for r in selected):
        raise ValueError('Diagnostics accept train videos only')
    tasks = [(r['video_id'], Path(r['video_path'])) for r in selected]
    args.output.mkdir(parents=True, exist_ok=False)
    detector = CashDetector(args.model)
    from PIL import Image
    blank = [Image.new('RGB', (640, 360), 'black') for _ in range(6)]
    times = [0, 2, 4, 6, 8, 10]
    text = PROMPT + json.dumps(times) + '\nSchema: ' + json.dumps(WindowDecision.model_json_schema())
    formatted = detector.template(detector.processor, detector.model.config, text, num_images=6)
    response = detector.generate(detector.model, detector.processor, formatted, image=blank,
                                 max_tokens=170, temperature=0.0, verbose=False)
    (args.output / 'synthetic-schema.txt').write_text(response.text, encoding='utf-8')
    try:
        decision = parse_decision(response.text, times)
        sentinel = {'synthetic_blank_decision': decision.decision}
    except ValueError:
        sentinel = {'synthetic_blank_decision': 'invalid_response'}
    (args.output / 'config.json').write_text(json.dumps({
        'prompt': GROUNDED_PROMPT, 'frames': 8, 'max_side': 640,
        'prompt_sha256': hashlib.sha256(GROUNDED_PROMPT.encode()).hexdigest(),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model': str(args.model.resolve()), 'split': 'train', **sentinel}, indent=2), encoding='utf-8')
    print(json.dumps(sentinel), flush=True)
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as file:
        for key, path in tasks:
            start = time.monotonic()
            frames, timestamps, _ = sample_video(path, 2, 640, start+240)
            indexes = sorted({round(i*(len(frames)-1)/7) for i in range(8)})
            images, times = [frames[i] for i in indexes], [timestamps[i] for i in indexes]
            formatted = detector.template(detector.processor, detector.model.config,
                                          GROUNDED_PROMPT + json.dumps(times), num_images=len(images))
            response = detector.generate(detector.model, detector.processor, formatted, image=images,
                                         max_tokens=280, temperature=0.0, verbose=False)
            (args.output / f'{key}-reasoning.txt').write_text(response.text, encoding='utf-8')
            try:
                prediction, status = parse_final(response.text), 'done'
            except ValueError:
                prediction, status = None, 'failed'
            result = {'video_id': key, 'prediction': prediction, 'status': status,
                      'elapsed_sec': round(time.monotonic()-start, 3), 'sampled_timestamps': times}
            file.write(json.dumps(result)+'\n')
            file.flush()
            print(json.dumps({k: result[k] for k in ['video_id', 'prediction', 'status', 'elapsed_sec']}), flush=True)
            detector.mx.clear_cache()


if __name__ == '__main__':
    main()
