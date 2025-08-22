import os
import subprocess
import pytest
from pathlib import Path

from libkernelbot.launchers import ModalLauncher
from libkernelbot.consts import SubmissionMode, GPU_TO_SM, ModalGPU
from libkernelbot.task import make_task_definition, build_task_config
from libkernelbot.report import RunProgressReporter


class MockProgressReporter(RunProgressReporter):
    """Test progress reporter that captures messages."""

    def __init__(self, title: str = "Test Modal Run"):
        super().__init__(title)
        self.messages = []
        self.updates = []

    async def push(self, message: str):
        self.messages.append(message)

    async def update(self, message: str):
        self.updates.append(message)


@pytest.fixture(scope="session")
def modal_deployment(project_root: Path):
    """
    Fixture that ensures Modal is deployed before running tests.
    Runs once per test session and deploys to the specified Modal environment.
    """
    # Determine Modal environment (default to 'test' if not specified)
    modal_env = os.getenv("PYTEST_MODAL_ENV", "pytest")

    print(f"🚀 Deploying to Modal environment: {modal_env}")

    # Deploy to Modal with specific environment
    try:
        result = subprocess.run(
            ["modal", "deploy", "--env", modal_env, "modal_runner_archs.py"],
            cwd=project_root / "src" / "runners",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout in case image needs to be built (can be very slow)
        )

        if result.returncode != 0:
            # if it fails simply because the environment does not exist, we can fix  that
            if "No such environment" in result.stderr:
                result = subprocess.run(
                    ["modal", "environment", "create", modal_env],
                    cwd=project_root / "src" / "runners",
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    pytest.fail(f"Modal environment `{modal_env}` not available, and failed to create: {result.stderr}")
                else:
                    # try again, now that the env exists.
                    result = subprocess.run(
                        ["modal", "deploy", "--env", modal_env, "modal_runner_archs.py"],
                        cwd=project_root / "src" / "runners",
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode != 0:
                        pytest.fail(f"Modal deploy failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            else:
                pytest.fail(f"Modal deploy failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        print(f"✅ Modal deployment to '{modal_env}' completed successfully")
        print(f"Deploy output: {result.stdout}")

        # Set the Modal environment for the session
        original_env = os.environ.get("MODAL_ENVIRONMENT")
        os.environ["MODAL_ENVIRONMENT"] = modal_env

        yield modal_env

        # Restore original environment
        if original_env is not None:
            os.environ["MODAL_ENVIRONMENT"] = original_env
        elif "MODAL_ENVIRONMENT" in os.environ:
            del os.environ["MODAL_ENVIRONMENT"]

    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Modal deploy timed out after 5 minutes:\nstdout: {e.stdout}, stderr:{e.stderr}")
    except Exception as e:
        pytest.fail(f"Modal deploy failed with exception: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_type", [ModalGPU.T4, ModalGPU.L4, ModalGPU.A100, ModalGPU.H100, ModalGPU.B200])
async def test_modal_launcher_python_script(modal_deployment, project_root: Path, gpu_type: ModalGPU):
    """
    Test ModalLauncher with a real Python script using examples/identity_py.
    """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")

    # Load the real identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping Modal integration test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / "submission.py").read_text()

    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[gpu_type.name],
        mode=SubmissionMode.TEST
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert isinstance(result.runs, dict)

    # System info - test actual expected values
    assert gpu_type.name in result.system.gpu
    assert "Linux" in result.system.platform
    assert result.system.torch.startswith("2.7")  # update when the image changes

    # Test run structure
    assert 'test' in result.runs
    test_run = result.runs['test']

    # For Python runs, compilation is None
    assert test_run.compilation is None

    # Run needs to succeed
    assert test_run.run.success is True
    assert test_run.run.passed is True
    assert test_run.run.exit_code == 0
    assert test_run.run.stdout == ""
    assert test_run.run.stderr == ""
    assert test_run.run.duration > 0

    # Test need to succeed
    assert test_run.run.result['check'] == 'pass'
    test_count = int(test_run.run.result['test-count'])
    assert test_count == 5
    for i in range(test_count):
        assert test_run.run.result[f'test.{i}.status'] == 'pass'
        assert 'size:' in test_run.run.result[f'test.{i}.spec']
        assert 'seed:' in test_run.run.result[f'test.{i}.spec']

    # sanity check for timings
    assert test_run.start < test_run.end

    # check messages
    assert reporter.messages == ['⏳ Waiting for Modal run to finish...']
    assert reporter.updates == ['✅ Waiting for modal run to finish... Done']


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["cheat-fd.py", "cheat-input.py", "cheat-rng.py"])
async def test_modal_launcher_failing_script(modal_deployment, project_root: Path, script: str):
    """
        Test ModalLauncher with a real Python scripts that are designed to be wrong.
        """
    launcher = ModalLauncher(add_include_dirs=[])
    reporter = MockProgressReporter("progress")
    gpu_type = ModalGPU.T4

    # Load the real identity_py task
    task_path = project_root / "examples" / "identity_py"
    if not task_path.exists():
        pytest.skip("examples/identity_py not found - skipping Modal integration test")

    # Load the task definition
    task_definition = make_task_definition(task_path)

    # Use the actual working submission from the examples
    submission_content = (task_path / script).read_text()
    task_definition.task.seed = 653212
    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_content,
        arch=GPU_TO_SM[gpu_type.name],
        mode=SubmissionMode.LEADERBOARD,
    )

    result = await launcher.run_submission(config, gpu_type, reporter)

    # Basic structure and success
    assert result.success, f"Expected successful run, got: {result.error}"
    assert result.error == ""
    assert result.runs['test'].run.passed is False or result.runs['benchmark'].run.passed is False
