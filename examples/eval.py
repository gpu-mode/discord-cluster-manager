import dataclasses
import re
import time
import os
import sys
import math
from pathlib import Path

import torch.cuda

from utils import set_seed
try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from submission import custom_kernel
from reference import check_implementation, generate_input

WARMUP_RUNS = 10
TIMED_RUNS = 100


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, 'w')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)
    
    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def get_test_cases(file_name: str) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    return tests


def warm_up(test: TestCase):
    data = generate_input(**test.args)
    start = time.perf_counter()
    while time.perf_counter() - start < 0.2:
        custom_kernel(data)
        torch.cuda.synchronize()


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg)**2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


def run_testing(logger: PopcornOutput, tests: list[TestCase]):
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)

        data = generate_input(**test.args)
        torch.cuda.synchronize()
        submission_output = custom_kernel(data)
        torch.cuda.synchronize()
        error = check_implementation(data, submission_output)
        if error:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", error)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float):
    durations = []
    # generate input data once
    data = generate_input(**test.args)
    #  first, one obligatory correctness check
    output = custom_kernel(data)
    error = check_implementation(data, output)
    if error:
        return error

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    for i in range(max_repeats):
        if recheck:
            data = generate_input(**test.args)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = custom_kernel(data)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        if recheck:
            error = check_implementation(data, output)
            if error:
                return error

        del output
        durations.append(end-start)

        if i > 1:
            stats = calculate_stats(durations)
            if stats.err / stats.mean < 0.01 or stats.mean *  stats.runs > max_time_ns:
                break

    return calculate_stats(durations)


def run_benchmarking(logger: PopcornOutput, tests: list[TestCase]):
    warm_up(tests[0])
    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = benchmark(test, False, 100, 10e9)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    tests = get_test_cases(sys.argv[2])

    with PopcornOutput(int(fd)) as logger:
        seed = os.getenv("POPCORN_SEED")
        seed = int(seed) if seed else 42
        set_seed(seed)

        if mode == "test":
            return run_testing(logger, tests)

        if mode == "benchmark":
            return run_benchmarking(logger, tests)

        # TODO implement leaderboard, script, and profile mode


if __name__ == "__main__":
    sys.exit(main())
