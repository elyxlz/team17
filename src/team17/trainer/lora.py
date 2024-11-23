"""https://github.com/cccntu/minLoRA"""

import math
import typing

import torch


class LoRAParametrization(torch.nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        rank: int = 4,
        lora_alpha: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, fan_in, dtype=dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(fan_out, rank, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = lora_alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.lora_B @ self.lora_A) * self.scaling


def add_lora(model: torch.nn.Module, lora_rank: int, skip: list[str] = []) -> None:
    for name, layer in model.named_modules():
        if not any([m in name for m in skip]):
            if isinstance(layer, torch.nn.Linear):
                fan_out, fan_in = layer.weight.shape
                torch.nn.utils.parametrize.register_parametrization(  # type: ignore
                    layer,
                    "weight",
                    LoRAParametrization(
                        fan_in, fan_out, rank=lora_rank, dtype=layer.weight.dtype
                    ),
                )


def merge_lora(model: torch.nn.Module) -> torch.nn.Module:
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    for layer in model.modules():
        if hasattr(layer, "parametrizations"):
            # Create a list of keys first before iteration
            param_keys = list(layer.parametrizations.keys())
            for attr_name in param_keys:
                torch.nn.utils.parametrize.remove_parametrizations(
                    layer, attr_name, leave_parametrized=True
                )
    return model


def get_lora_params(
    model: torch.nn.Module,
) -> typing.Iterator[tuple[str, torch.nn.Parameter]]:
    for name, param in model.named_parameters():
        if "lora" in name:
            yield name, param


def get_lora_state_dict(model: torch.nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora" in k}
