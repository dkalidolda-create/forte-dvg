"""Train-only visual demonstrations packed into temporal panels; inference is local."""
import hashlib
import json
import re
import time

from ml.grounded_cash import GroundedCashDetector, choose_indexes, parse_label
from ml.local_cash import sample_video

PROMPT = '''Three images contain chronological panels from security-camera videos.
Within each panel read left to right, then top to bottom. Each small image is
labelled with its timestamp. The FIRST TWO panels are labelled training examples.
Their known labels illustrate the specific category, not legal guilt.
Category A: accepted banknotes remain on the open cashier work surface after the
cash-handling operation ends and the cashier switches away from that operation.
Category B: this event is not observed, e.g. ongoing counting/service or storage.
Category U: the relevant objects or temporal transition cannot be resolved.
Do not identify people, infer authorization, or treat paper documents as money.
The THIRD panel is the QUERY. Classify only that panel, not the examples.
Return ONE LETTER ONLY: B, U, or A.
Training labels in panel order: '''


def montage(frames, timestamps, indexes):
    from PIL import Image, ImageDraw
    width, height = max(frames[i].width for i in indexes), max(frames[i].height for i in indexes)
    columns = 2
    canvas = Image.new('RGB', (columns*width, ((len(indexes)+1)//2)*(height+24)), '#202020')
    draw = ImageDraw.Draw(canvas)
    for number, i in enumerate(indexes):
        x, y = number % columns * width, number // columns * (height+24)
        canvas.paste(frames[i], (x, y+24))
        draw.text((x+5, y+4), f'{timestamps[i]:.3f} sec', fill='white')
    return canvas


class ExampleCashDetector(GroundedCashDetector):
    def __init__(self, model_path, examples=None, **kwargs):
        if not examples or sorted(target for _, target in examples) != [0, 1]:
            raise ValueError('Exactly one positive and one negative train example required')
        super().__init__(model_path, **kwargs)
        self.support = []
        self.support_labels = []
        supports = []
        for path, target in sorted(examples, key=lambda pair: pair[1]):
            frames, times, _ = sample_video(path, 2, self.max_side, time.monotonic()+60)
            indexes = choose_indexes(len(frames), 4)
            self.support.append(montage(frames, times, indexes))
            self.support_labels.append('A' if target else 'B')
            supports.append({'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'target': target})
        self.config.update({'detector': 'example_panels_v1', 'query_frames': 8, 'support_frames': 4,
                            'demonstrations': supports,
                            'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest()})

    def analyze(self, video_path, log_dir):
        start = time.monotonic()
        frames, times, duration = sample_video(video_path, 2, self.max_side, start+self.deadline)
        indexes = choose_indexes(len(frames), self.config['query_frames'])
        query = montage(frames, times, indexes)
        images = self.support+[query]
        log_dir.mkdir(parents=True, exist_ok=False)
        answer = self.ask(PROMPT+json.dumps(self.support_labels), images, 8)
        (log_dir / 'decision.txt').write_text(answer, encoding='utf-8')
        prediction = parse_label(answer)
        events = []
        if prediction == 1:
            query_times = [times[i] for i in indexes]
            answer = self.ask('Consider ONLY the third (QUERY) panel. Select the timestamp '
                              'that most clearly supports cash left after the operation. '
                              'Return ONE NUMBER from '+json.dumps(query_times)+', without explanation.', images, 16)
            (log_dir / 'localization.txt').write_text(answer, encoding='utf-8')
            if not re.fullmatch(r'\s*\d+(?:\.\d+)?\s*', answer):
                raise ValueError('Invalid event timestamp')
            selected = float(answer)
            nearest = min(query_times, key=lambda t: abs(t-selected))
            if not abs(nearest-selected) <= .51:
                raise ValueError('Event timestamp outside query frames')
            events = [nearest]
        elapsed = time.monotonic()-start
        if elapsed > self.deadline:
            raise TimeoutError('analysis_timeout')
        result = {'prediction': prediction, 'event_times_sec': events,
                  'sampled_frames': len(indexes), 'sampled_timestamps': [times[i] for i in indexes],
                  'duration_sec': round(duration, 3), 'elapsed_sec': round(elapsed, 3),
                  'peak_memory_gb': round(self.mx.get_peak_memory()/1e9, 3)}
        (log_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
