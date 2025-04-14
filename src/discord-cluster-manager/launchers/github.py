import asyncio
import datetime
import json
import pprint
import tempfile
import zipfile
from typing import Awaitable, Callable, Optional

import requests
from consts import AMD_REQUIREMENTS, GPU, NVIDIA_REQUIREMENTS, GitHubGPU, GPUType
from github import Github, UnknownObjectException, WorkflowRun
from report import RunProgressReporter
from run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo
from utils import get_github_branch_name, setup_logging

from .launcher import Launcher

logger = setup_logging()


class GitHubLauncher(Launcher):
    def __init__(self, repo: str, token: str):
        super().__init__(name="GitHub", gpus=GitHubGPU)
        self.repo = repo
        self.token = token

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA

        lang = config["lang"]
        if lang == "cu" and selected_gpu == GPUType.AMD:
            # TODO implement HIP
            raise NotImplementedError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        if selected_gpu == GPUType.AMD:
            gpu_name = config.get("gpu", "mi300")
            runner_name = {"mi250": "amdgpu-mi250-x86-64", "mi300": "amdgpu-mi300-x86-64"}[gpu_name]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {selected_gpu.name}")
        if selected_gpu == GPUType.AMD:
            logger.info(f"Running on {gpu_name} amd gpu")

        workflow_file = selected_gpu.value
        run = GitHubRun(self.repo, self.token, workflow_file)

        payload = json.dumps(config)

        inputs = {"payload": payload}
        if lang == "py":
            if selected_gpu == GPUType.NVIDIA:
                inputs["requirements"] = NVIDIA_REQUIREMENTS
            else:
                inputs["requirements"] = AMD_REQUIREMENTS
                inputs["runner"] = runner_name
        if not await run.trigger(inputs):
            raise RuntimeError("Failed to trigger GitHub Action. Please check the configuration.")

        await status.push("⏳ Waiting for workflow to start...")
        await run.wait_for_completion(lambda x: self.wait_callback(x, status))
        await status.update(f"Workflow [{run.run_id}]({run.html_url}) completed")
        await status.push("Downloading artifacts...")

        artifacts = await run.download_artifacts()
        if "run-result" not in artifacts:
            logger.error("Could not find `run-result` among artifacts: %s", artifacts.keys())
            await status.push("Downloading artifacts...  failed")
            return FullResult(
                success=False, error="Could not download artifacts", runs={}, system=SystemInfo()
            )

        logs = artifacts["run-result"]["result.json"].decode("utf-8")

        await status.update("Downloading artifacts... done")

        data = json.loads(logs)
        runs = {}
        # convert json back to EvalResult structures, which requires
        # special handling for datetime and our dataclasses.
        for k, v in data["runs"].items():
            if "compilation" in v and v["compilation"] is not None:
                comp = CompileResult(**v["compilation"])
            else:
                comp = None
            run = RunResult(**v["run"])
            res = EvalResult(
                start=datetime.datetime.fromisoformat(v["start"]),
                end=datetime.datetime.fromisoformat(v["end"]),
                compilation=comp,
                run=run,
            )
            runs[k] = res

        system = SystemInfo(**data.get("system", {}))
        return FullResult(success=True, error="", runs=runs, system=system)

    async def wait_callback(self, run: "GitHubRun", status: RunProgressReporter):
        await status.update(
            f"⏳ Workflow [{run.run_id}]({run.html_url}): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )


class GitHubRun:
    def __init__(self, repo: str, token: str, workflow_file: str):
        gh = Github(token)
        self.repo = gh.get_repo(repo)
        self.token = token
        self.workflow_file = workflow_file
        self.run: Optional[WorkflowRun.WorkflowRun] = None
        self.start_time = None

    @property
    def run_id(self):
        if self.run is None:
            return None
        return self.run.id

    @property
    def html_url(self):
        if self.run is None:
            return None
        return self.run.html_url

    @property
    def status(self):
        if self.run is None:
            return None
        return self.run.status

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return None
        return datetime.datetime.now(datetime.timezone.utc) - self.start_time

    async def trigger(self, inputs: dict) -> bool:
        """
        Trigger this run with the provided inputs.
        Sets `self.run` to the new WorkflowRun on success.

        Returns: Whether the run was successfully triggered,
        """
        trigger_time = datetime.datetime.now(datetime.timezone.utc)
        try:
            workflow = self.repo.get_workflow(self.workflow_file)
        except UnknownObjectException as e:
            logger.error(f"Could not find workflow {self.workflow_file}", exc_info=e)
            raise ValueError(f"Could not find workflow {self.workflow_file}") from e

        logger.debug(
            "Dispatching workflow %s on branch %s with inputs %s",
            self.workflow_file,
            get_github_branch_name(),
            pprint.pformat(inputs),
        )
        success = workflow.create_dispatch(get_github_branch_name(), inputs=inputs)
        if success:
            await asyncio.sleep(2)
            runs = list(workflow.get_runs())

            for run in runs:
                if run.created_at.replace(tzinfo=datetime.timezone.utc) > trigger_time:
                    self.run = run
                    return True
        return False

    async def wait_for_completion(
        self, callback: Callable[["GitHubRun"], Awaitable[None]], timeout_minutes: int = 5
    ):
        if self.run is None:
            raise ValueError("Run needs to be triggered before a status check!")

        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        timeout = datetime.timedelta(minutes=timeout_minutes)

        while True:
            try:
                # update run status
                self.run = run = self.repo.get_workflow_run(self.run_id)

                if self.elapsed_time > timeout:
                    try:
                        self.run.cancel()
                        # Wait briefly to ensure cancellation is processed
                        # And Verify the run was actually cancelled
                        await asyncio.sleep(5)
                        run = self.repo.get_workflow_run(self.run_id)
                        if run.status != "completed":
                            logger.warning(f"Failed to cancel workflow run {self.run_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling workflow: {str(e)}", exc_info=e)
                        raise

                    logger.warning(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )
                    raise TimeoutError(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )

                if run.status == "completed":
                    return

                await callback(self)
                await asyncio.sleep(20)
            except TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error waiting for GitHub run {self.run_id}: {e}", exc_info=e)
                raise

    async def download_artifacts(self) -> dict:
        logger.info("Attempting to download artifacts for run %s", self.run_id)
        artifacts = self.run.get_artifacts()

        extracted = {}

        for artifact in artifacts:
            url = artifact.archive_download_url
            headers = {"Authorization": f"token {self.token}"}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile("w+b") as temp:
                    temp.write(response.content)
                    temp.flush()

                    with zipfile.ZipFile(temp.name) as z:
                        artifact_dict = {}
                        for file in z.namelist():
                            with z.open(file) as f:
                                artifact_dict[file] = f.read()

                extracted[artifact.name] = artifact_dict
            else:
                raise RuntimeError(
                    f"Failed to download artifact {artifact.name}. "
                    f"Status code: {response.status_code}"
                )

        logger.info("Download artifacts for run %s: %s", self.run_id, list(extracted.keys()))
        return extracted
