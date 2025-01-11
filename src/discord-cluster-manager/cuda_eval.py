from consts import CUDA_FLAGS, MODAL_CUDA_INCLUDE_DIRS
from timeout import TimeoutException, timeout


def run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
    arch: int = None,
) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        timeout_seconds: Maximum execution time in seconds (default: 600 seconds)
        arch: The arch code for the compute/sm versions.

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import os
    import subprocess
    import time

    try:
        with timeout(timeout_seconds):
            # Check CUDA is available and installed correctly
            print("[CUDA Env Check]")
            try:
                # these check cuda compiler is also available
                subprocess.run(["nvcc", "--version"], check=True)
                subprocess.run(["which", "nvcc"], check=True)
            except Exception:
                return "nvcc not found.", 0.0

            ARCH = f"-gencode=arch=compute_{arch},code=sm_{arch}"
            NVCC_FILES = "eval.cu"
            # Write submission files to directory
            if reference_content is not None:
                with open("reference.cuh", "w") as f:
                    f.write(reference_content)

            if submission_content is not None:
                with open("train.cuh", "w") as f:
                    f.write(submission_content)

            with open("eval.cu", "w") as f:
                f.write(script_content)

            execution_start_time = time.perf_counter()
            compile_process = subprocess.run(
                ["nvcc"]
                + CUDA_FLAGS
                + MODAL_CUDA_INCLUDE_DIRS
                + [ARCH, NVCC_FILES, "-o", "eval.out"],
                capture_output=True,
                text=True,
            )

            if compile_process.returncode != 0:
                raise RuntimeError(
                    "CUDA compilation failed with return code "
                    + f"{compile_process.returncode}:\n{compile_process.stderr}"
                )

            run_process = subprocess.run(["./eval.out"], capture_output=True, text=True)
            execution_end_time = time.perf_counter()

            print("run process stdout", run_process.stdout)

            score = None
            for line in run_process.stdout.splitlines():
                if line.startswith("score:"):
                    score = float(line.split(":")[1].strip())
                    break

            if score is None:
                execution_end_time = time.perf_counter()
                score = execution_end_time - execution_start_time
                return (
                    "check_implementation failed"
                    if "check_implementation failed" in run_process.stdout
                    else None,
                    score,
                )  # To make sure error is thrown on LB

            return run_process.stdout, score

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["reference.cuh", "train.cuh", "eval.cu", "eval.out"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
