"""Local frozen visual encoder + supervised crop classifier. No text generation."""
import gc
import os
import time

import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

CLASSES = ['banknotes', 'documents', 'cash_storage']


def normalize(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


class FrozenVisionFeatures:
    def __init__(self, path):
        import mlx.core as mx
        from mlx_vlm import load

        started = time.monotonic()
        model, self.processor = load(str(path.resolve()), trust_remote_code=False)
        self.vision = model.vision_tower
        self.mx = mx
        del model
        gc.collect()
        mx.clear_cache()
        self.load_sec = time.monotonic()-started

    def encode(self, image):
        mx = self.mx
        batch = self.processor.image_processor.preprocess(
            images=[image.convert('RGB')], min_pixels=65536, max_pixels=65536)
        pixels = mx.array(batch['pixel_values']).astype(self.vision.patch_embed.proj.weight.dtype)
        grid = mx.array(batch['image_grid_thw'])
        final, intermediate = self.vision(pixels, grid)
        pooled = [mx.mean(layer.astype(mx.float32), axis=0) for layer in [final, *intermediate]]
        mx.eval(*pooled)
        vector = normalize(np.concatenate([normalize(np.asarray(layer)) for layer in pooled]))
        mx.clear_cache()
        return vector.astype(np.float32)


class RidgeCropClassifier:
    """Balanced kernel-ridge classifier with fixed regularization (not calibrated probabilities)."""
    def __init__(self, classes=None):
        self.classes = list(classes or CLASSES)

    def fit(self, vectors, targets, alpha=.1):
        self.train = normalize(np.asarray(vectors, dtype=np.float64))
        targets = np.asarray(targets, dtype=int)
        counts = np.bincount(targets, minlength=len(self.classes))
        if not np.all(counts):
            raise ValueError('All object classes required in training')
        root_weights = np.sqrt(len(targets)/(len(self.classes)*counts[targets]))
        kernel = self.train@self.train.T
        # Constant feature permits an intercept; regularization remains fixed.
        kernel += 1
        weighted = root_weights[:, None]*kernel*root_weights[None, :]
        self.dual = root_weights[:, None]*np.linalg.solve(
            weighted+alpha*np.eye(len(targets)), root_weights[:, None]*np.eye(len(self.classes))[targets])
        return self

    def scores(self, vectors):
        return (normalize(np.asarray(vectors))@self.train.T+1)@self.dual

    def predict(self, vectors):
        return self.scores(vectors).argmax(axis=1)

    def save(self, path):
        np.savez(path, train=self.train, dual=self.dual, classes=np.array(self.classes))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        classes = data['classes'].tolist()
        if len(set(classes)) != len(classes) or not set(classes) <= set(CLASSES+['collection_bag']):
            raise ValueError('Unknown classifier classes')
        instance = cls(classes)
        instance.train, instance.dual = data['train'], data['dual']
        return instance
