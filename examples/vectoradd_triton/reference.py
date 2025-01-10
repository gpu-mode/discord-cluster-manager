import torch
from typing import List, Tuple

def generate_input() -> List[List[torch.Tensor]]:
    configs = [
        (1024, 1024),
        (1024, 2048),
        (2048, 2048),
        (4096, 4096),
    ]
    
    return [[
        torch.randn(M, N, device='cuda', dtype=torch.float16).contiguous(),
        torch.randn(M, N, device='cuda', dtype=torch.float16).contiguous()
    ] for M, N in configs]

def ref_kernel(inputs: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    return [A + B for A, B in inputs]

def check_implementation(
    custom_outputs: List[torch.Tensor],
    reference_outputs: List[torch.Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> bool:
    if len(custom_outputs) != len(reference_outputs):
        return False
    
    for i, (custom_output, reference_output) in enumerate(zip(custom_outputs, reference_outputs)):
        if custom_output.shape != reference_output.shape:
            return False
        
        if not torch.allclose(custom_output, reference_output, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(custom_output - reference_output)).item()
            return False
    
    return True
