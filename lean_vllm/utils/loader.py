import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# Routed experts are stored one by one, and load into one stacked parameter per projection.
EXPERT_WEIGHT = re.compile(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
STACKED_EXPERT_PARAMS = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj"}


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if expert := EXPERT_WEIGHT.fullmatch(weight_name):
                    prefix, expert_id, proj = expert.groups()
                    param = model.get_parameter(f"{prefix}.{STACKED_EXPERT_PARAMS[proj]}")
                    param.weight_loader(param, f.get_tensor(weight_name), (int(expert_id), proj))
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
