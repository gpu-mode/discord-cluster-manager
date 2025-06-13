import torch
from task import input_t, output_t
from utils import verbose_allclose
from typing import Tuple

def generate_input(size: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    data = torch.empty(size, device='cuda', dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    return data


def ref_kernel(data: input_t) -> output_t:
    return data


def check_implementation(data, output) -> Tuple[bool, str]:
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)
    if len(reasons) > 0:
        # TODO better processing of reasons
        return False, "mismatch found! custom implementation doesn't match reference.: " + reasons[0]

    return True, ''


