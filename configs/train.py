from transformers.convert_slow_tokenizer import generate_merges
from team17.trainer.train import MyUltravoxTrainConfig, train
import os

import wandb.util

os.environ["RUN_ID"] = wandb.util.generate_id()


config = MyUltravoxTrainConfig(
    seed=42,
    run_name="blabla",
    # Model path
    ultravox_pretrained_path="i2xmortal/Llama-3.2-300M-untrained",
    text_tokenizer_path="i2xmortal/Llama-3.2-300M-untrained",
    # ultravox_pretrained_path="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    # text_tokenizer_path="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    # ultravox_kwargs={"num_hidden_layers": 1, "hidden_size": 64, "vocab_size": 128},
    # Text configuration
    text_max_length=128,
    # Audio configuration
    sample_rate=24000,
    # LoRA parameters
    lora_r=8,
    lora_alpha=16,
    # Training flags
    fake=True,
    overfit=False,
    compile=False,
    gradient_checkpointing=True,
    # Data loading
    batch_size=2,
    num_workers=4,
    test_size=2,
    generate_kwargs={},
    # Training parameters
    lr=5e-5,
    warmup_steps=100,
    max_steps=10000,
    # Monitoring intervals
    test_every=500,
    save_every=None,
    push_every=None,
    watch_every=None,
)

# Start training
train(config)
