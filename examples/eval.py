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
        tests.append(case)

    return tests


def run_test(idx: int, test: TestCase, logger: PopcornOutput):
    logger.log(f"test.{idx}.spec", test.spec)

    data = generate_input(**test.args)
    submission_output = custom_kernel(data)
    error = check_implementation(data, submission_output)
    if error:
        logger.log(f"test.{idx}.status", "fail")
        logger.log(f"test.{idx}.error", error)
        return True
    else:
        logger.log(f"test.{idx}.status", "pass")
        return False


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


def benchmark(idx: int, test: TestCase, logger: PopcornOutput):
    durations = []
    # generate input data once
    data = generate_input(**test.args)

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    for i in range(100):
        start = time.perf_counter_ns()
        output = custom_kernel(data)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        del output
        durations.append(end-start)

        if i > 1:
            stats = calculate_stats(durations)
            if stats.err / stats.mean < 0.01 or stats.mean *  stats.runs > 10e9:
                break

    stats = calculate_stats(durations)
    logger.log(f"duration.{idx}.spec", str(test))
    for field in dataclasses.fields(Stats):
        logger.log(f"duration.{idx}.{field.name}", getattr(stats, field.name))


def measure_for_leaderboard(test: TestCase, logger: PopcornOutput):


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

        passed = True
        if mode == "test" or mode == "benchmark":
            for idx, test in enumerate(tests):
                if run_test(idx, test, logger):
                    passed = False
                    if mode == "benchmark":
                        break

        if passed:
            logger.log("check", "pass")
        else:
            logger.log("check", "fail")
            return 112

        if mode == "test":
            return 0

        if mode == "benchmark":
            warm_up(tests[0])
            for idx, test in enumerate(tests):
                benchmark(idx, test, logger)
            return 0

        if mode == "leaderboard":
            warm_up(tests[0])

        auto result = measure_for_leaderboard(logger, tests.back(), seed);
        logger.log("duration.spec", tests.back().spec.c_str());
        logger.log("duration.runs", result.runs);
        logger.log("duration.mean", result.mean);
        logger.log("duration.std", result.std);
        logger.log("duration.err", result.err);
        logger.log("duration.best", result.best);
        logger.log("duration.worst", result.worst);
        } else {
        std::
            cerr << "Unknown evaluation mode '" << mode << "'" << std::endl;
        return ExitCodes::USAGE_ERROR;
        }

        return 2


if __name__ == "__main__":
    sys.exit(main())
