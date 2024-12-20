import os

import wandb.util

from team17.trainer.train import MyUltravoxTrainConfig, train

os.environ["RUN_ID"] = wandb.util.generate_id()

testing = os.getenv("TESTING") == "True"

config = MyUltravoxTrainConfig(
    seed=42,
    run_name="blabla",
    # Model path
    ultravox_pretrained_path="fixie-ai/ultravox-v0_3-llama-3_2-1b"
    if not testing
    else None,
    # Audio configuration
    sample_rate=24000,
    # LoRA parameters
    lora_r=16,
    lora_alpha=16,
    # Training flags
    compile=False,
    gradient_checkpointing=True,
    # Data loading
    batch_size=32 if not testing else 2,
    num_workers=8 if not testing else 0,
    test_size=2,
    generate_kwargs={},
    # Training parameters
    lr=5e-5,
    warmup_steps=20,
    max_steps=500,
    # Monitoring intervals
    test_every=None,
    save_every=None,
    push_every=None,
    watch_every=None,
)

# Start training
train(config)
