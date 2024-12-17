import asyncio
import os
import zipfile
from datetime import datetime, timedelta, timezone

import discord
import requests
from consts import GITHUB_REPO, GITHUB_TOKEN, GPUType
from discord import app_commands
from discord.ext import commands
from github import Github
from leaderboard_eval import cu_eval, py_eval
from utils import get_github_branch_name, send_discord_message, setup_logging

logger = setup_logging()


class GitHubCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.run_github = bot.run_group.command(
            name="github", description="Run a script using GitHub Actions"
        )(self.run_github)

    @app_commands.describe(
        script="The Python script file to run",
        gpu_type="Choose the GPU type for GitHub Actions",
    )
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name="NVIDIA", value="nvidia"),
            app_commands.Choice(name="AMD", value="amd"),
        ]
    )
    async def run_github(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        reference_script: discord.Attachment = None,
        reference_code: str = None,
    ) -> discord.Thread:
        if not script.filename.endswith(".py") and not script.filename.endswith(".cu"):
            await send_discord_message(
                interaction, "Please provide a Python (.py) or CUDA (.cu) file"
            )
            return None

        thread = await self.bot.create_thread(interaction, gpu_type.name, "GitHub Job")
        message = f"Created thread {thread.mention} for your GitHub job"

        await send_discord_message(interaction, message)
        await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

        try:
            script_content = (await script.read()).decode("utf-8")
            selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA

            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )
                eval_code = py_eval if script.filename.endswith(".py") else cu_eval

                run_id = await self.trigger_github_action(
                    script_content,
                    script.filename,
                    selected_gpu,
                    reference_content,
                    eval_code,
                )
            else:
                run_id = await self.trigger_github_action(
                    script_content, script.filename, selected_gpu
                )

            if run_id:
                await thread.send(
                    f"GitHub Action triggered! Run ID: {run_id}\nMonitoring progress..."
                )
                status, logs, url = await self.check_workflow_status(run_id, thread)

                await thread.send(f"Training completed with status: {status}")

                if len(logs) > 1900:
                    await self.bot.send_chunked_message(thread, logs, code_block=True)
                else:
                    await thread.send(f"```\nLogs:\n{logs}\n```")

                if url:
                    await thread.send(f"View the full run at: <{url}>")
            else:
                await thread.send(
                    "Failed to trigger GitHub Action. Please check the configuration."
                )

            return thread

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                await thread.send(f"Error processing request: {str(e)}")
            raise

    async def trigger_github_action(
        self,
        script_content,
        filename,
        gpu_type,
        reference_content=None,
        eval_content=None,
    ):
        logger.info(f"Attempting to trigger GitHub action for {gpu_type.name} GPU")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            trigger_time = datetime.now(timezone.utc)
            workflow_file = gpu_type.value
            workflow = repo.get_workflow(workflow_file)

            if reference_content is not None:
                eval_filename = "eval.py" if filename.endswith(".py") else "eval.cu"
                reference_filename = (
                    "reference.py" if filename.endswith(".py") else "reference.cu"
                )
                success = workflow.create_dispatch(
                    get_github_branch_name(),
                    {
                        "script_content": script_content,
                        "filename": filename,
                        "reference_content": reference_content,
                        "reference_filename": reference_filename,
                        "eval_content": eval_content,
                        "eval_filename": eval_filename,
                    },
                )
            else:
                success = workflow.create_dispatch(
                    get_github_branch_name(),
                    {"script_content": script_content, "filename": filename},
                )

            if success:
                await asyncio.sleep(2)
                runs = list(workflow.get_runs())

                for run in runs:
                    if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                        return run.id
            return None

        except Exception as e:
            logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
            return None

    async def check_workflow_status(self, run_id, thread):
        logger.info(f"Starting to monitor workflow status for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)
        start_time = datetime.now(timezone.utc)
        timeout_minutes = 5
        timeout = timedelta(minutes=timeout_minutes)

        while True:
            try:
                run = repo.get_workflow_run(run_id)
                elapsed_time = datetime.now(timezone.utc) - start_time

                if elapsed_time > timeout:
                    try:
                        run.cancel()
                        # Wait briefly to ensure cancellation is processed
                        # And Verify the run was actually cancelled
                        await asyncio.sleep(5)
                        run = repo.get_workflow_run(run_id)
                        if run.status != "completed":
                            logger.warning(f"Failed to cancel workflow run {run_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling workflow: {str(e)}")

                    await thread.send(
                        f"Workflow cancelled - exceeded {timeout_minutes} minute timeout"
                    )
                    return (
                        "cancelled",
                        f"Workflow exceeded {timeout_minutes} minute timeout",
                        run.html_url,
                    )

                if run.status == "completed":
                    logs = await self.download_artifact(run_id)
                    return run.conclusion, logs, run.html_url

                await thread.send(
                    f"Workflow: {run.status} running for {elapsed_time.total_seconds():.2f} seconds\n"
                    f"Live view: <{run.html_url}>"
                )
                await asyncio.sleep(60)
            except Exception as e:
                return "error", str(e), None

    async def download_artifact(self, run_id):
        logger.info(f"Attempting to download artifacts for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            run = repo.get_workflow_run(run_id)
            artifacts = run.get_artifacts()

            for artifact in artifacts:
                if artifact.name == "training-artifacts":
                    url = artifact.archive_download_url
                    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                    response = requests.get(url, headers=headers)

                    if response.status_code == 200:
                        with open("training.log.zip", "wb") as f:
                            f.write(response.content)

                        with zipfile.ZipFile("training.log.zip") as z:
                            log_file = next(
                                (f for f in z.namelist() if f.endswith("training.log")),
                                None,
                            )
                            if log_file:
                                with z.open(log_file) as f:
                                    logs = f.read().decode("utf-8")
                            else:
                                logs = "training.log file not found in artifact"

                        os.remove("training.log.zip")
                        return logs
                    else:
                        return f"Failed to download artifact. Status code: {response.status_code}"

            return "No training artifacts found"
        except Exception as e:
            return f"Error downloading artifacts: {str(e)}"
