########
# Evaluation scripts to run for leaderboard results
########

py_eval = """
import torch
import time
from reference import ref_kernel, generate_input
from train import custom_kernel


def check_implementation():
    for _ in range(10):  # check multiple times
        input_tensors = generate_input()
        for input in input_tensors:
            custom_output = custom_kernel(input, dim=-1)
            ref_output = ref_kernel(input, dim=-1)

            if not torch.allclose(custom_output, ref_output, atol=1e-5):
                print('mismatch found! custom implementation doesn't match reference.')
                return

    print('custom implementation matches the reference implementation.')


def metric():
    warmup_runs = 10
    timed_runs = 100

    # warmup
    print('warming up...')
    for _ in range(warmup_runs):
        input_tensors = generate_input()
        for input in input_tensors:
            _ = custom_kernel(input, dim=-1)
            _ = ref_kernel(input, dim=-1)

    # timing
    print('timing custom implementation...')
    input_tensor = generate_input()
    start_time = time.time()
    for _ in range(timed_runs):
        for input in input_tensors:
            _ = custom_kernel(input, dim=-1)

    custom_duration = (time.time() - start_time) / timed_runs

    print(f'submitted kernel runtime: {custom_duration:.4f} seconds')

    return custom_duration

def main():
    check_implementation()
    s = metric()

    print(f'score:{s}')

if __name__ == '__main__':
    main()

"""

cu_eval = """

"""
