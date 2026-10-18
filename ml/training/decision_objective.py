"""Process-local MLX trainer extension; installed dependencies stay unchanged."""
from ml.training.label_mask import label_weights


def install():
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy
    from mlx_vlm.trainer.vlm.sft import runtime
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset

    original = VisionDataset.process

    def process(self, row):
        batch = original(self, row)
        content = self._conversation(row)[-1]['content']
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        weights = label_weights(tokenizer, content, np.asarray(batch['input_ids']).tolist())
        batch['objective_weights'] = mx.array(weights, dtype=mx.float32)
        return batch

    def loss(model, batch, **_kwargs):
        batch = dict(batch)
        weights = batch.pop('objective_weights')
        ids, attention = batch['input_ids'], batch['attention_mask']
        padding = ids.shape[1]-weights.shape[1]
        if padding < 0:
            raise ValueError('Supervision is longer than model input')
        weights = mx.pad(weights, [(0, 0), (0, padding)])[:, 1:]*attention[:, 1:]
        logits = runtime.forward_vlm_logits(model, ids[:, :-1], attention[:, :-1], batch)
        targets = ids[:, 1:]
        losses = token_cross_entropy(logits, targets, ignore_index=None)
        total = mx.maximum(weights.sum(), 1e-8)
        objective = (losses*weights).sum()/total
        accuracy = ((mx.argmax(logits, axis=-1) == targets)*weights).sum()/total
        metrics = {'weight': total, 'num_tokens': (weights > 0).sum(),
                   'token_accuracy': accuracy, 'nll': objective,
                   'num_vision_tokens': runtime.vlm_vision_mask(model, batch).sum()}
        return objective, {**metrics, **runtime.vlm_batch_metrics(batch)}

    VisionDataset.process = process
    runtime.make_task.__kwdefaults__['loss_fn'] = loss
