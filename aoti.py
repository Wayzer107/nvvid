"""
"""

from typing import cast

import torch
from huggingface_hub import hf_hub_download
from spaces.zero.torch.aoti import ZeroGPUCompiledModel
from spaces.zero.torch.aoti import ZeroGPUWeights
from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters


def aoti_blocks_load(module: torch.nn.Module, repo_id: str, variant: str | None = None):
    repeated_blocks = cast(list[str], module._repeated_blocks)
    aoti_files = {name: hf_hub_download(
        repo_id=repo_id,
        filename='package.pt2',
        subfolder=name if variant is None else f'{name}.{variant}',
    ) for name in repeated_blocks}
    for block_name, aoti_file in aoti_files.items():
        for block in module.modules():
            if block.__class__.__name__ == block_name:
                block_ = _shallow_clone_module(block)
                unwrap_tensor_subclass_parameters(block_)
                weights = ZeroGPUWeights(block_.state_dict())
                block.forward = ZeroGPUCompiledModel(aoti_file, weights)
