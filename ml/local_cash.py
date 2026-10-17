"""Experimental local cash-event detector. No ground-truth input or cloud calls."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

PROMPT = '''Inspect these chronological security-camera frames of a cashier desk.
Detect ONLY this event: a customer transaction visibly ends (the customer leaves,
or the cashier starts serving another customer) AND the already accepted banknotes
visibly remain outside a cash-storage device, on the work surface, AFTER that end.
Cash on a desk DURING counting or an ongoing transaction is NOT this event.
Do not assume transaction completion merely because the cashier pauses or looks away.
Do not identify people. Do not infer permissions, missing documents, or legal guilt.
If necessary details are too small or the temporal context is ambiguous, use uncertain.
Use present only when all three evidence flags are true. Use absent when this event
is not observed and uncertain when a plausible event cannot be resolved visually.
Return ONLY a JSON object matching the supplied schema, without descriptions or markdown.
For present, event_time_sec must be one of the supplied timestamps AFTER completion,
showing cash left outside storage. Never return a frame index. For absent or uncertain
use null. Do not copy example values: decide each evidence flag from the images.
Frame timestamps: '''


class WindowDecision(BaseModel):
    model_config = ConfigDict(strict=True)
    decision: Literal['present', 'absent', 'uncertain']
    operation_end_visible: bool
    cash_visible_after_end: bool
    outside_storage: bool
    event_time_sec: Optional[float]

    @model_validator(mode='after')
    def consistent(self):
        flags = self.operation_end_visible and self.cash_visible_after_end and self.outside_storage
        if self.decision == 'present' and (not flags or self.event_time_sec is None):
            raise ValueError('Positive decision without supporting flags/frame')
        if self.decision == 'absent' and flags:
            raise ValueError('Absent decision contradicts all supporting flags')
        if self.decision != 'present' and self.event_time_sec is not None:
            raise ValueError('Non-positive decision must not select an event frame')
        if self.event_time_sec is not None and self.event_time_sec < 0:
            raise ValueError('Negative timestamp')
        return self


def parse_decision(text, timestamps):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    decision = WindowDecision.model_validate_json(text)
    if decision.event_time_sec is not None:
        nearest = min(timestamps, key=lambda t: abs(t-decision.event_time_sec))
        if not abs(nearest-decision.event_time_sec) <= .51:
            raise ValueError('Event timestamp outside sampled frames')
        decision.event_time_sec = nearest
    return decision


def sample_video(video_path, interval, max_side, deadline):
    """Sequential decoding avoids unreliable seeking past the final decodable frame."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    frames, timestamps = [], []
    last_frame, last_time = None, 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('invalid_video')
        index, next_time = 0, 0.0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('analysis_timeout')
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp = pts if index == 0 or pts > last_time else index / fps
            if last_frame is not None and timestamp < last_time:
                timestamp = last_time + 1/fps
            last_frame, last_time = frame, timestamp
            if timestamp + 1e-6 >= next_time:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((max_side, max_side))
                frames.append(image)
                timestamps.append(round(timestamp, 3))
                next_time += interval
            index += 1
        if last_frame is None:
            raise ValueError('invalid_video')
        if not timestamps or last_time - timestamps[-1] > .05:
            image = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((max_side, max_side))
            frames.append(image)
            timestamps.append(round(last_time, 3))
        return frames, timestamps, last_time + 1/fps
    finally:
        cap.release()


class CashDetector:
    def __init__(self, model_path: Path, interval=2.0, frames_per_window=6,
                 stride=3, max_side=640, deadline_sec=240, examples=None):
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        if not model_path.is_dir():
            raise ValueError('Local model directory does not exist')
        if interval <= 0 or not 1 <= stride <= frames_per_window or frames_per_window < 2:
            raise ValueError('Invalid sampling parameters')
        self.mx, self.generate, self.template = mx, generate, apply_chat_template
        started = time.monotonic()
        self.model, self.processor = load(str(model_path.resolve()), trust_remote_code=False)
        self.load_sec = time.monotonic() - started
        self.interval, self.window, self.stride = interval, frames_per_window, stride
        self.max_side, self.deadline = max_side, deadline_sec
        self.example_images, self.example_text, example_info = [], '', []
        for number, (path, target) in enumerate(examples or []):
            images, times, _ = sample_video(path, interval, max_side, time.monotonic()+60)
            indexes = sorted({0, len(images)//3, 2*len(images)//3, len(images)-1})
            self.example_images.extend(images[i] for i in indexes)
            self.example_text += (f'Demonstration {number+1}: next {len(indexes)} images '
                                  f'at seconds {[times[i] for i in indexes]}; known category '
                                  f'label={"present" if target else "absent"}. ')
            example_info.append({'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'target': target, 'frames': len(indexes)})
        self.config = {'model': str(model_path.resolve()), 'interval_sec': interval,
                       'window_frames': frames_per_window, 'stride_frames': stride,
                       'max_side': max_side, 'deadline_sec': deadline_sec,
                       'demonstrations': example_info,
                       'prompt_sha256': hashlib.sha256((PROMPT + json.dumps(WindowDecision.model_json_schema(), sort_keys=True)).encode()).hexdigest()}

    def analyze(self, video_path: Path, log_dir: Path):
        """Accept video ONLY. Truth labels and annotated timestamps are not inputs."""
        start = time.monotonic()
        frames, timestamps, duration = sample_video(video_path, self.interval, self.max_side,
                                                   start+self.deadline)
        log_dir.mkdir(parents=True, exist_ok=False)
        events, results = [], []
        offsets = list(range(0, max(1, len(frames) - self.window + 1), self.stride))
        last_start = max(0, len(frames) - self.window)
        if offsets[-1] != last_start:
            offsets.append(last_start)
        for number, offset in enumerate(offsets):
            if time.monotonic() - start > self.deadline:
                raise TimeoutError('analysis_timeout')
            images = frames[offset:offset + self.window]
            times = timestamps[offset:offset + self.window]
            prefix = (self.example_text + f'QUERY: only the final {len(images)} images '
                      'are the video window to classify. Do not report demonstration events. '
                      if self.example_images else '')
            input_images = self.example_images + images
            prompt = self.template(self.processor, self.model.config,
                                   prefix + PROMPT + json.dumps(times) + '\nSchema: '
                                   + json.dumps(WindowDecision.model_json_schema()), num_images=len(input_images))
            response = self.generate(self.model, self.processor, prompt, image=input_images,
                                     max_tokens=170, temperature=0.0, verbose=False)
            text = response.text if hasattr(response, 'text') else str(response)
            (log_dir / f'window-{number:03d}.txt').write_text(text, encoding='utf-8')
            decision = parse_decision(text, times)
            results.append({'timestamps': times, **decision.model_dump()})
            if decision.decision == 'present':
                timestamp = decision.event_time_sec
                if not events or timestamp - events[-1] >= 4:
                    events.append(timestamp)
            self.mx.clear_cache()
        elapsed = time.monotonic() - start
        if elapsed > self.deadline:
            raise TimeoutError('analysis_timeout')
        uncertain = sum(r['decision'] == 'uncertain' for r in results)
        prediction = 1 if events else (None if uncertain else 0)
        result = {'prediction': prediction, 'event_times_sec': events,
                  'uncertain_windows': uncertain, 'windows': len(results),
                  'sampled_frames': len(frames), 'duration_sec': round(duration, 3),
                  'elapsed_sec': round(elapsed, 3),
                  'peak_memory_gb': round(self.mx.get_peak_memory() / 1e9, 3)}
        (log_dir / 'windows.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        (log_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
