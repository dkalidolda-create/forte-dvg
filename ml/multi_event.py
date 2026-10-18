"""Local six-frame visual event adapter. No labels or source IDs at inference."""
import os
import time
from pathlib import Path

from ml.event_schema import parse, prompt
from ml.grounded_cash import choose_indexes
from ml.local_cash import sample_video

for _name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK']:
    os.environ[_name] = '1'


class EventDetector:
    def __init__(self, model_path, adapter_path=None):
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self.mx, self.generate, self.template = mx, generate, apply_chat_template
        mx.set_memory_limit(9*1024**3)
        mx.set_cache_limit(512*1024**2)
        started = time.monotonic()
        self.model, self.processor = load(str(Path(model_path).resolve()),
                                         adapter_path=str(Path(adapter_path).resolve()) if adapter_path else None,
                                         trust_remote_code=False)
        self.load_sec = round(time.monotonic()-started, 3)

    def classify(self, images, times):
        formatted = self.template(self.processor, self.model.config, prompt(times), num_images=len(images))
        response = self.generate(self.model, self.processor, formatted, image=images,
                                 max_tokens=300, temperature=0.0, verbose=False)
        self.mx.clear_cache()
        return parse(response.text, len(images))

    def decision_margin(self, images, times):
        """Score the learned binary decision before generating JSON boilerplate.

        The threshold must be fitted on training examples only and evaluated
        separately. This is not a calibrated probability or a legal confidence.
        """
        tokenizer = self.processor.tokenizer
        positive = tokenizer.encode('violation', add_special_tokens=False)[0]
        negative = tokenizer.encode('no_violation', add_special_tokens=False)[0]
        if positive == negative:
            raise ValueError('Non-distinct decision tokens')
        values = []

        def record(_tokens, logits):
            if not values:
                delta = logits.reshape(-1)[positive]-logits.reshape(-1)[negative]
                self.mx.eval(delta)
                values.append(float(delta))
            return logits

        formatted = self.template(self.processor, self.model.config, prompt(times), num_images=len(images))
        self.generate(self.model, self.processor, formatted+'{"verdict": "', image=images,
                      max_tokens=1, temperature=0.0, logits_processors=[record], verbose=False)
        self.mx.clear_cache()
        if not values:
            raise ValueError('No binary decision score')
        return values[0]

    def classify_at_threshold(self, images, times, threshold):
        margin = self.decision_margin(images, times)
        if margin < threshold:
            return {'verdict': 'no_violation', 'events': []}, margin
        prefix = '{"verdict": "violation", "events": ['
        formatted = self.template(self.processor, self.model.config, prompt(times), num_images=len(images))
        response = self.generate(self.model, self.processor, formatted+prefix, image=images,
                                 max_tokens=280, temperature=0.0, verbose=False)
        self.mx.clear_cache()
        return parse(prefix+response.text, len(images)), margin

    def analyze(self, video_path):
        started = time.monotonic()
        frames, times, duration = sample_video(video_path, 1., 448, started+230)
        indexes = choose_indexes(len(frames), 6)
        selected_times = [times[i] for i in indexes]
        result = self.classify([frames[i] for i in indexes], selected_times)
        return {**result, 'sampled_timestamps': selected_times, 'duration_sec': duration,
                'elapsed_sec': round(time.monotonic()-started, 3)}
