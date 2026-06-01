from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LORA_WEIGHTS = Path(__file__).parent / "lora_weights" / "clip_lora.pt"
DEFAULT_TARGET_MODULES = ("mlp.c_fc", "mlp.c_proj")


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base = self.base_layer(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base + update * self.scale


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.0,
               target_modules=DEFAULT_TARGET_MODULES):
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def lora_state_dict(model: nn.Module):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_a" in name or ".lora_b" in name
    }


def save_lora(model: nn.Module, output_path, rank: int, alpha: int, dropout: float,
              target_modules=DEFAULT_TARGET_MODULES):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": tuple(target_modules),
            "state_dict": lora_state_dict(model),
        },
        output_path,
    )


def load_lora(model: nn.Module, weights_path=DEFAULT_LORA_WEIGHTS, map_location="cpu"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return False

    checkpoint = torch.load(weights_path, map_location=map_location)
    apply_lora(
        model,
        rank=checkpoint["rank"],
        alpha=checkpoint["alpha"],
        dropout=checkpoint.get("dropout", 0.0),
        target_modules=checkpoint.get("target_modules", DEFAULT_TARGET_MODULES),
    )
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    if unexpected_lora:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected_lora}")
    return True


def require_lora(model: nn.Module, weights_path=DEFAULT_LORA_WEIGHTS, map_location="cpu"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LoRA weights not found: {weights_path}. "
            "Download clip_lora.pt and place it under lora_weights/ before running."
        )
    load_lora(model, weights_path=weights_path, map_location=map_location)
    return True
