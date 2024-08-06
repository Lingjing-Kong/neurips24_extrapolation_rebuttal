
import torch.nn as nn
import math
import os
from typing import Optional, List, Dict, Union
from functools import partial
from copy import copy

import torch
import torch.nn as nn
from safetensors.torch import save_file


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        lora_dim_factor=0,
    ):
        super().__init__()
        self.lora_name = lora_name

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

            self.lora_dim = lora_dim if lora_dim_factor == 0 else max(1, max(in_dim, out_dim) // lora_dim_factor)

            self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

           
            self.lora_dim = min(lora_dim, in_dim, out_dim) if lora_dim_factor == 0 else max(1, max(in_dim, out_dim) // lora_dim_factor)

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        alpha = lora_dim 
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  
        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward


    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rank: Union[int, List] = 4,
        multiplier: float = 1.0,

    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.model = model
        self.module = partial(GATINGLoRAModule, multiplier=multiplier, lora_dim=rank, lora_dim_factor=0)

     

        self.loras = self.create_modules(self.model)
        print(f"create LoRA: {len(self.loras)} modules.")


        lora_names = set()
        for lora in self.loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)


        for lora in self.loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )


    def create_modules(
        self,
        root_module: nn.Module,
        prefix: str = "lora",

    ) -> list:
        loras = nn.ModuleList([])
        names = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d"]:
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")
                lora = self.module(lora_name, module)
                if lora_name not in names:
                    loras.append(lora)
                    names.append(lora_name)
        return loras
    
    def forward(self, x, **kwargs):
        with self:
            output = self.model(x, **kwargs)
        return output

    def prepare_optimizer_params(self):
        lora_params = [
            p for lora in self.loras for n, p in lora.named_parameters()  if "org" not in n
        ]
      
        backbone_params = [p for p in self.model.parameters()]
        
        return lora_params, backbone_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.loras:
            lora.multiplier = 0

    
######### lora gating ###################
class GATINGLoRAModule(LoRAModule):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        lora_dim_factor=1,
    ):
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            lora_dim_factor=lora_dim_factor,
        )
        
        if "Linear" in org_module.__class__.__name__:
            self.lora_gate = nn.Parameter(torch.randn(1, self.lora_dim))    
        elif "Conv" in org_module.__class__.__name__:  
            self.lora_gate = nn.Parameter(torch.randn(1, self.lora_dim, 1, 1))
        else:
            raise NotImplementedError


    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x) * self.lora_gate) * self.multiplier * self.scale
        )
    
    



