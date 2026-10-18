"""Local bounded-memory LoRA training using the installed MLX-VLM trainer."""
import os

for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK']:
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    import mlx.core as mx
    from mlx_vlm.train import main as train

    # Avoid exhausting the 16GB laptop while the web service is open.
    mx.set_memory_limit(9*1024**3)
    mx.set_cache_limit(512*1024**2)
    if os.getenv('EVENT_LABEL_OBJECTIVE') == '1':
        from ml.training.decision_objective import install
        install()
        print('Objective: decision=1.0/category=0.25; JSON/frame tokens excluded', flush=True)
    train()


if __name__ == '__main__':
    main()
