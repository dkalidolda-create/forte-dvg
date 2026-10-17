"""Experimental learned cash crops + workstation occupancy. Fully local inference."""
import hashlib
import json
import time

import numpy as np

from ml.dino_crops import DinoFeatures
from ml.grounded_cash import GroundedCashDetector, choose_indexes
from ml.local_cash import sample_video
from ml.vision_crops import RidgeCropClassifier, normalize

ROLE_PROMPT = '''These are chronological frames of a cashier workplace.
Assess ONLY whether staff occupy the cashier's working area, using body position
and the workspace, not identity or faces. A customer on the other side of the
counter does not count as staff at the workstation. A cashier standing nearby,
turning to storage, or moving within reach of the desk still counts as present.
For EACH supplied frame return one integer: 0 = staff present in the working area;
1 = staff have left the area and the workstation is unattended; 2 = cannot tell.
Do not judge banknotes, paperwork or legal violations. Do not assume a pause
means departure. Return ONLY a JSON object mapping each supplied frame ID to its integer code.
Include the first frame too. Use the exact supplied IDs as quoted keys.
If a frame cannot be assessed use code 2. Do not shift answers between frames.
Frame IDs and chronological timestamps: '''


def parse_occupancy(text, count):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    values = json.loads(text)
    if not isinstance(values, list) or len(values) != count or any(type(v) is not int or v not in [0, 1, 2] for v in values):
        raise ValueError('Invalid workstation occupancy array')
    return values


def confirmed_indexes(cash, occupancy):
    return [i for i in range(len(cash)-1)
            if cash[i] and cash[i+1] and occupancy[i] == occupancy[i+1] == 1]


def parse_indexed_occupancy(text, count):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    values = json.loads(text)
    allowed = {str(i) for i in range(count)}
    if (not isinstance(values, dict) or not values or not set(values) <= allowed
            or any(type(v) is not int or v not in [0, 1, 2] for v in values.values())):
        raise ValueError('Invalid indexed workstation occupancy')
    return [values.get(str(i), 2) for i in range(count)]


def proposals(annotations):
    if annotations.get('split') != 'train':
        raise ValueError('Proposal locations must use train annotations only')
    boxes = [b for row in annotations['annotations'] for b in row['boxes'] if b['object'] == 'banknotes']
    points = np.array([[b['x']+b['width']/2, b['y']+b['height']/2, b['width'], b['height']] for b in boxes])
    if len(points) < 4:
        raise ValueError('Insufficient manual banknote locations')
    centers = [points[0]]
    while len(centers) < 4:
        distance = ((points[:, None]-np.array(centers)[None])**2).sum(axis=2).min(axis=1)
        centers.append(points[distance.argmax()])
    centers = np.array(centers)
    for _ in range(12):
        assigned = ((points[:, None]-centers[None])**2).sum(axis=2).argmin(axis=1)
        centers = np.array([np.median(points[assigned == i], axis=0) if (assigned == i).any() else centers[i]
                            for i in range(len(centers))])
    result = []
    for cx, cy, width, height in centers:
        for scale in [1.0, 1.4]:
            for dx, dy in [(0, 0), (-.3, 0), (.3, 0), (0, -.3), (0, .3)]:
                x, y = cx+dx*width, cy+dy*height
                result.append([max(0., x-width*scale/2), max(0., y-height*scale/2),
                               min(1., x+width*scale/2), min(1., y+height*scale/2)])
    return result


class FastCashDetector(GroundedCashDetector):
    def __init__(self, model_path, annotations, crop_model, dino_model, **kwargs):
        super().__init__(model_path, **kwargs)
        self.encoder = DinoFeatures(dino_model)
        self.classifier = RidgeCropClassifier.load(crop_model/'classifier.npz')
        data = json.loads(annotations.read_text())
        self.boxes = proposals(data)
        support = np.load(crop_model/'support.npz', allow_pickle=False)
        self.cash_support = normalize(support['vectors'][support['targets'] == 0])
        self.config.update({'detector': 'fast_cash_v1', 'frames': 8,
                            'proposal_boxes': self.boxes, 'minimum_cash_similarity': .80,
                            'minimum_class_margin': .15, 'minimum_consecutive_unattended': 2,
                            'dino_model': str(dino_model.resolve()), 'crop_model': str(crop_model.resolve()),
                            'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
                            'crop_weights_sha256': hashlib.sha256((crop_model/'classifier.npz').read_bytes()).hexdigest(),
                            'prompt_sha256': hashlib.sha256(ROLE_PROMPT.encode()).hexdigest(),
                            'scope': 'Exposed cash persisting after staff departure; fixed camera proposals. '
                                     'Does not cover every kind of cash-left-after-operation event.'})

    def scan(self, image, deadline):
        features = []
        for x1, y1, x2, y2 in self.boxes:
            if time.monotonic() > deadline:
                raise TimeoutError('analysis_timeout')
            crop = image.crop((int(x1*image.width), int(y1*image.height),
                               max(int(x1*image.width)+1, int(x2*image.width)),
                               max(int(y1*image.height)+1, int(y2*image.height))))
            features.append(self.encoder.encode(crop))
        features = np.stack(features)
        scores = self.classifier.scores(features)
        cash_index = self.classifier.classes.index('banknotes')
        margins = scores[:, cash_index]-np.delete(scores, cash_index, axis=1).max(axis=1)
        similarity = (normalize(features)@self.cash_support.T).max(axis=1)
        matches = (margins >= .15) & (similarity >= .80)
        best = int(margins.argmax())
        return {'cash_detected': bool(matches.any()), 'matching_proposals': int(matches.sum()),
                'best_box': self.boxes[best], 'best_margin': float(margins[best]),
                'best_similarity': float(similarity[best])}

    def analyze(self, video_path, log_dir):
        start = time.monotonic()
        frames, times, duration = sample_video(video_path, 2, 1920, start+self.deadline)
        indexes = choose_indexes(len(frames), 8)
        chosen, timestamps = [frames[i] for i in indexes], [times[i] for i in indexes]
        log_dir.mkdir(parents=True, exist_ok=False)
        scan_start = time.monotonic()
        scans = [self.scan(image, start+self.deadline) for image in chosen]
        scan_sec = time.monotonic()-scan_start
        views = [image.copy() for image in chosen]
        for image in views:
            image.thumbnail((448, 448))
        frame_map = {str(i): t for i, t in enumerate(timestamps)}
        answer = self.ask(getattr(self, 'phase_prompt', ROLE_PROMPT)+json.dumps(frame_map), views, 128)
        (log_dir/'occupancy-answer.txt').write_text(answer, encoding='utf-8')
        occupancy = parse_indexed_occupancy(answer, len(chosen))
        cash = [r['cash_detected'] for r in scans]
        confirmed = confirmed_indexes(cash, occupancy)
        if confirmed:
            prediction = 1
        elif any(c and state in [1, 2] for c, state in zip(cash, occupancy)):
            prediction = None
        else:
            prediction = 0
        (log_dir/'scans.json').write_text(json.dumps({'timestamps': timestamps,
                        'scans': scans, 'occupancy': occupancy}, indent=2), encoding='utf-8')
        elapsed = time.monotonic()-start
        if elapsed > self.deadline:
            raise TimeoutError('analysis_timeout')
        result = {'prediction': prediction, 'event_times_sec': [timestamps[confirmed[0]]] if confirmed else [],
                  'sampled_frames': len(chosen), 'duration_sec': round(duration, 3),
                  'cash_scan_sec': round(scan_sec, 3), 'elapsed_sec': round(elapsed, 3),
                  'peak_memory_gb': round(self.mx.get_peak_memory()/1e9, 3)}
        (log_dir/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
