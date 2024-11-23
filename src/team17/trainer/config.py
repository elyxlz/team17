import pydantic as pyd

from team17.trainer import utils


class MyUltravoxTrainConfig(utils.BaseConfig):
    run_id: str = pyd.Field(init=False)
    seed: int
    run_name: str

    ## model
    ultravox_pretrained_path: str | None

    sample_rate: int

    ## lora
    lora_r: int
    lora_alpha: int

    ## data
    data_path: str = "./data/tokens"
    batch_size: int
    num_workers: int
    test_size: int

    ## speed
    compile: bool
    gradient_checkpointing: bool

    ## optimizer
    lr: float
    warmup_steps: int
    max_steps: int

    ## test
    test_every: int | None
    generate_kwargs: dict

    ## logging and checkpointing
    save_every: int | None
    push_every: int | None
    watch_every: int | None
    ckpt_path: str | None = None
    wandb_project_name: str = "ultravox"
