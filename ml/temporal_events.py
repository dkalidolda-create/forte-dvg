"""Small, local supervised event classifier on chronological visual features.

Runtime accepts image-derived vectors only. Training videos are few and share a
camera; agreement on them is training replay, not evidence of generalization.
"""
import numpy as np

from ml.vision_crops import normalize


def kernel(left, right, gamma=8.0):
    left = normalize(np.asarray(left, dtype=np.float64))
    right = normalize(np.asarray(right, dtype=np.float64))
    return np.exp(-gamma*np.maximum(0.0, 2-2*(left@right.T)))+1.0


class TemporalEventHead:
    def __init__(self, support, coefficients, categories, time_coefficients, time_masks,
                 alpha=0.01, gamma=8.0):
        self.support = np.asarray(support)
        self.coefficients = np.asarray(coefficients)
        self.categories = list(categories)
        self.time_coefficients = np.asarray(time_coefficients)
        self.time_masks = np.asarray(time_masks, dtype=bool)
        self.alpha, self.gamma = float(alpha), float(gamma)
        n, c = len(self.support), len(self.categories)
        if (self.support.ndim != 2 or self.coefficients.shape != (n, c+1)
                or self.time_coefficients.shape != (n, c) or self.time_masks.shape != (n, c)
                or not all(np.isfinite(x).all() for x in
                           [self.support, self.coefficients, self.time_coefficients])):
            raise ValueError('Invalid visual head weights')

    @classmethod
    def fit(cls, vectors, binary, category_targets, normalized_times, categories,
            alpha=0.01, gamma=8.0):
        support = normalize(np.asarray(vectors, dtype=np.float64))
        binary = np.asarray(binary, dtype=int)
        targets = np.asarray(category_targets, dtype=int)
        times = np.asarray(normalized_times, dtype=float)
        if (alpha <= 0 or set(binary.tolist()) != {0, 1} or targets.shape != times.shape
                or targets.shape != (len(binary), len(categories))
                or np.any((targets != 0) & (targets != 1))
                or np.any((targets.sum(axis=1) > 0) != binary.astype(bool))):
            raise ValueError('Invalid training targets')
        gram = kernel(support, support, gamma)
        labels = np.column_stack([binary, targets])*2-1
        coefficients = np.linalg.solve(gram+alpha*np.eye(len(binary)), labels)
        timing = np.zeros_like(targets, dtype=float)
        masks = targets.astype(bool)
        for category in range(len(categories)):
            indexes = np.flatnonzero(masks[:, category])
            if not len(indexes):
                continue
            timing[indexes, category] = np.linalg.solve(
                gram[np.ix_(indexes, indexes)]+alpha*np.eye(len(indexes)), times[indexes, category])
        return cls(support, coefficients, categories, timing, masks, alpha, gamma)

    def score(self, vectors):
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        if values.shape[1] != self.support.shape[1] or not np.isfinite(values).all():
            raise ValueError('Invalid visual features')
        similarity = kernel(values, self.support, self.gamma)
        return similarity@self.coefficients, similarity@self.time_coefficients

    def predict(self, vector):
        scores, times = self.score(vector)
        scores, times = scores[0], times[0]
        if scores[0] < 0:
            return {'verdict': 'no_violation', 'events': [], 'raw_margin': float(scores[0])}
        selected = np.flatnonzero(scores[1:] >= 0).tolist()
        # A binary positive still needs the most likely learned event type. This
        # is a classifier decision, not a verified finding or calibrated confidence.
        if not selected:
            selected = [int(np.argmax(scores[1:]))]
        return {'verdict': 'violation', 'events': [
            {'category': self.categories[i], 'normalized_time': float(np.clip(times[i], 0, 0.999))}
            for i in selected], 'raw_margin': float(scores[0])}

    def save(self, path):
        np.savez_compressed(path, support=self.support.astype(np.float32),
                            coefficients=self.coefficients, categories=np.asarray(self.categories),
                            time_coefficients=self.time_coefficients, time_masks=self.time_masks,
                            alpha=self.alpha, gamma=self.gamma)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as value:
            return cls(**{key: value[key] for key in
                          ['support', 'coefficients', 'categories', 'time_coefficients', 'time_masks', 'alpha', 'gamma']})
