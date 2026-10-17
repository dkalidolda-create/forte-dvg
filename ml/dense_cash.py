"""Experimental native-resolution tile search plus temporal operation review, offline."""
import hashlib
import time

import numpy as np

from ml.dino_crops import DinoFeatures
from ml.fast_cash import FastCashDetector
from ml.grounded_cash import GroundedCashDetector
from ml.vision_crops import RidgeCropClassifier, normalize

PHASE_PROMPT = '''These are chronological views of a cashier workplace.
For EACH supplied frame ID assess the visible stage of the cash transaction.
0 = cash handling or customer service is still ongoing;
1 = the prior interaction visibly ended, service visibly switched to another
customer, or the employee left the workstation;
2 = cannot establish the stage from the images.
Standing nearby or turning to storage is not leaving. Paperwork, a pause, or an
empty-looking customer area alone does not establish completion. Use chronological
context. Do not identify anyone, infer ownership, or make legal judgments.
Do not classify the money itself. Return ONLY a JSON object mapping all supplied
frame IDs to integers 0, 1, or 2. Missing evidence must be code 2.
Frame IDs and chronological timestamps: '''


class DenseCashDetector(FastCashDetector):
    def __init__(self, model_path, annotations, crop_model, dino_model, **kwargs):
        GroundedCashDetector.__init__(self, model_path, **kwargs)
        self.encoder = DinoFeatures(dino_model)
        self.classifier = RidgeCropClassifier.load(crop_model/'classifier.npz')
        support = np.load(crop_model/'support.npz', allow_pickle=False)
        self.cash_support = normalize(support['vectors'][support['targets'] == 0])
        self.config.update({'detector': 'dense_cash_v2', 'frames': 8,
                            'tile_size': 224, 'tile_stride': 112,
                            'minimum_cash_similarity': .80, 'minimum_class_margin': .15,
                            'minimum_consecutive_confirmations': 2,
                            'dino_model': str(dino_model.resolve()), 'crop_model': str(crop_model.resolve()),
                            'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
                            'crop_weights_sha256': hashlib.sha256((crop_model/'classifier.npz').read_bytes()).hexdigest(),
                            'prompt_sha256': hashlib.sha256(PHASE_PROMPT.encode()).hexdigest(),
                            'scope': 'Visible banknotes persisting after visible completion, service switch or departure. '
                                     'Experimental full-frame patch search; no unlabeled background supervision.'})
        self.phase_prompt = PHASE_PROMPT

    def scan(self, image, deadline):
        matches = []
        best_margin, best_similarity = -float('inf'), 0.
        best_box = [0., 0., 1., 1.]
        for features, left, top in self.encoder.tiles(image):
            if time.monotonic() > deadline:
                raise TimeoutError('analysis_timeout')
            scores = self.classifier.scores(features)
            cash_index = self.classifier.classes.index('banknotes')
            margins = scores[:, cash_index]-np.delete(scores, cash_index, axis=1).max(axis=1)
            similarity = (normalize(features)@self.cash_support.T).max(axis=1)
            for p in np.flatnonzero((margins >= .15) & (similarity >= .80)):
                x, y = left+(int(p)%16)*14, top+(int(p)//16)*14
                if x < image.width and y < image.height:
                    matches.append([x/image.width, y/image.height,
                                    min(1., (x+14)/image.width), min(1., (y+14)/image.height)])
            best = int(margins.argmax())
            if margins[best] > best_margin:
                best_margin, best_similarity = float(margins[best]), float(similarity[best])
                x, y = left+(best%16)*14, top+(best//16)*14
                best_box = [x/image.width, y/image.height,
                            min(1., (x+14)/image.width), min(1., (y+14)/image.height)]
        # Adjacent patches are correlated and not independent detections.
        return {'cash_detected': bool(matches), 'matching_proposals': len(matches),
                'best_box': best_box, 'best_margin': best_margin,
                'best_similarity': best_similarity, 'candidate_patch_boxes': matches}
