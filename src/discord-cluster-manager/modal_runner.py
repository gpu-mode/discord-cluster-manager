import subprocess
from typing import Optional

from consts import MODAL_PATH
from modal import App, Image, Mount
from timeout import TimeoutException, timeout

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
mount = Mount.from_local_dir(
    MODAL_PATH,
    remote_path="/root/",
)
app = App("discord-bot-runner")
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Move this to another file later:
python_image = Image.debian_slim(python_version="3.10").pip_install(
    ["torch", "triton", "jax[cuda12]", "jax2torch"]
)

cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "gcc-11",
        "g++-11",
        "clang-11",  # note i skip a step
    )
    .pip_install(
        "ninja", "packaging", "wheel", "torch", "numpy", "triton", "jax[cuda12]", "jax2torch"
    )
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 "
        + "--slave /usr/bin/g++ g++ /usr/bin/g++-11",
        # "apt update",
        # "apt  -y install clang-10", # this should be clang-10 but I can't get it to work yet
        #
        "git clone https://github.com/HazyResearch/ThunderKittens.git",
        # "cd /ThunderKittens && pwd && python setup.py install",
    )
)


def run_pytorch_script(  # noqa: C901
    script_content: str,
    reference_content: Optional[str] = None,
    submission_content: Optional[str] = None,
    timeout_seconds: int = 300,
    arch: int = None,
) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        timeout_seconds: Maximum execution time before timeout (default: 300 seconds)
        arch: The arch code for the compute/sm versions.

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """

    import os
    import time

    try:
        with timeout(timeout_seconds):
            # Write submission files to directory
            if reference_content is not None:
                with open("reference.py", "w") as f:
                    f.write(reference_content)

            if submission_content is not None:
                with open("train.py", "w") as f:
                    f.write(submission_content)

            with open("eval.py", "w") as f:
                f.write(script_content)

            execution_start_time = time.perf_counter()
            result = subprocess.run(
                ["python", "eval.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Script execution failed with return code "
                    + f"{result.returncode}:\n{result.stderr}"
                )

            score = None
            for line in result.stdout.splitlines():
                if line.startswith("score:"):
                    score = float(line.split(":")[1].strip())
                    return ("score", score)

            if score is None:
                execution_end_time = time.perf_counter()
                score = execution_end_time - execution_start_time

        return result.stdout, score

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["eval.py", "reference.py", "train.py"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
