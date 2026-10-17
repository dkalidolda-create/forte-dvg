"""Offline MLX startup/speed probe. Outputs are saved locally, never uploaded.

This is an experiment, NOT the worker analyzer and NOT a validated detector.
"""
import argparse
import json
import os
import time
from pathlib import Path

# Download weights separately. Inference must not contact Hugging Face or telemetry.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=4)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise ValueError('Download model weights first')
    if args.output.exists():
        raise ValueError('Output already exists; choose a new run directory')
    if not 1 <= args.frames <= 8:
        raise ValueError('Use 1 to 8 frames for the initial memory probe')

    import cv2
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image, ImageDraw

    args.output.mkdir(parents=True)
    images, timestamps = [], []
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total <= 0:
                raise ValueError('Invalid video')
            for index in range(args.frames):
                frame_index = min(total - 1, int((index + .5) * total / args.frames))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Cannot decode sampled frame')
                timestamp = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((448, 448))
                images.append(image)
                timestamps.append(round(timestamp, 3))
        finally:
            cap.release()
        prompt = (
            'These are sparse video frames in chronological order, at seconds '
            + json.dumps(timestamps)
            + '. Describe visible actions only, without identifying people. '
            'Do not infer legal violations, permissions, missing documents, or '
            'transaction completion from missing evidence. '
            'Return JSON only: {"observations": [{"timestamp_sec": number, '
            '"action": "brief visible action"}], "limitations": "brief"}.'
        )
    else:
        image = Image.new('RGB', (224, 224), 'white')
        ImageDraw.Draw(image).rectangle((50, 50, 170, 170), fill='red')
        images.append(image)
        prompt = 'What color is the rectangle? Answer in one word.'

    started = time.monotonic()
    model, processor = load(str(args.model.resolve()), trust_remote_code=False)
    loaded = time.monotonic()
    formatted = apply_chat_template(processor, model.config, prompt, num_images=len(images))
    result = generate(model, processor, formatted, image=images,
                      max_tokens=180 if args.video else 12, temperature=0.0, verbose=False)
    finished = time.monotonic()
    response = result.text if hasattr(result, 'text') else str(result)
    metrics = {'model': str(args.model), 'video': str(args.video) if args.video else None,
               'sampled_timestamps': timestamps, 'frames': len(images),
               'load_sec': round(loaded - started, 3),
               'inference_sec': round(finished - loaded, 3),
               'total_sec': round(finished - started, 3),
               'peak_memory_gb': round(mx.get_peak_memory() / 1e9, 3),
               'warning': 'Sparse-frame speed probe, not an accuracy evaluation or full-video analysis.'}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.output / 'response.txt').write_text(response, encoding='utf-8')
    # Deliberately do not print video-derived model descriptions into an external chat.
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
