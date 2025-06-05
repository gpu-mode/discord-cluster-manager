import copy
import math
from typing import TYPE_CHECKING, Optional

from launchers import Launcher

if TYPE_CHECKING:
    from bot import ClusterBot

import discord
from consts import GPU, GPU_TO_SM, RankCriterion, SubmissionMode, get_gpu_by_name
from discord import app_commands
from discord.ext import commands
from report import (
    MultiProgressReporter,
    RunProgressReporter,
    generate_report,
    make_short_report,
)
from run_eval import FullResult
from task import LeaderboardTask
from utils import (
    KernelBotError,
    build_task_config,
    send_discord_message,
    setup_logging,
    with_error_handling,
)

logger = setup_logging()


class SubmitCog(commands.Cog):
    """
    Code submission / run schedular cogs.

    Actual submission logic is handled by the launcher object.
    """

    def __init__(self, bot):
        self.bot: ClusterBot = bot
        self.launcher_map = {}

    def register_launcher(self, launcher: Launcher):
        choices = [app_commands.Choice(name=c.name, value=c.value) for c in launcher.gpus]

        run_fn = self.run_script

        # note: these helpers want to set custom attributes on the function, but `method`
        # does not allow setting any attributes, so we define this wrapper
        async def run(
            interaction: discord.Interaction,
            script: discord.Attachment,
            gpu_type: app_commands.Choice[str],
        ):
            return await run_fn(interaction, script, gpu_type)

        run = app_commands.choices(gpu_type=choices)(run)
        run = app_commands.describe(
            script="The Python/CUDA script file to run",
            gpu_type=f"Choose the GPU type for {launcher.name}",
        )(run)

        # For now, direct (non-leaderboard) submissions are debug-only.
        if self.bot.debug_mode:
            self.bot.run_group.command(
                name=launcher.name.lower(), description=f"Run a script using {launcher.name}"
            )(run)

        for gpu in launcher.gpus:
            self.launcher_map[gpu.value] = launcher

    async def submit_leaderboard(  # noqa: C901
        self,
        submission_id: int,
        code: str,
        name: str,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        task: LeaderboardTask,
        mode: SubmissionMode,
        seed: Optional[int],
    ) -> Optional[FullResult]:
        """
        Function invoked by `leaderboard_cog` to handle a leaderboard run.
        """
        if seed is not None:
            # careful, we've got a reference here
            # that is shared with the other run
            # invocations.
            task = copy.copy(task)
            task.seed = seed

        result = await self._handle_submission(
            gpu_type,
            reporter,
            code=code,
            name=name,
            task=task,
            mode=mode,
            submission_id=submission_id,
        )

        if result.success:
            score = None
            # Calculate score for both leaderboard and milestone runs
            score_run_key = None
            if "leaderboard" in result.runs and result.runs["leaderboard"].run.success and result.runs["leaderboard"].run.passed:
                score_run_key = "leaderboard"
            elif "milestone" in result.runs and result.runs["milestone"].run.success and result.runs["milestone"].run.passed:
                score_run_key = "milestone"
            
            if score_run_key:
                score = 0.0
                num_benchmarks = int(result.runs[score_run_key].run.result["benchmark-count"])
                if task.ranking_by == RankCriterion.LAST:
                    if num_benchmarks != 1:
                        logger.error(
                            "Ranked submission error for submission %d ranking_by is `last`, "
                            "but got %d benchmarks",
                            submission_id,
                            num_benchmarks,
                        )
                        raise KernelBotError(
                            f"Expected submission to have exactly one benchmark,"
                            f"got {num_benchmarks}."
                        )
                    score = float(result.runs[score_run_key].run.result["benchmark.0.mean"]) / 1e9
                else:
                    scores = []
                    for i in range(num_benchmarks):
                        scores.append(
                            float(result.runs[score_run_key].run.result[f"benchmark.{i}.mean"])
                            / 1e9
                        )
                    if task.ranking_by == RankCriterion.MEAN:
                        score = sum(scores) / len(scores)
                    elif task.ranking_by == RankCriterion.GEOM:
                        score = math.pow(math.prod(scores), 1.0 / num_benchmarks)

            # verifyruns uses a fake submission id of -1
            if submission_id != -1:
                with self.bot.leaderboard_db as db:
                    for key, value in result.runs.items():
                        # Assign score for leaderboard and milestone runs
                        run_score = None
                        if key == "leaderboard" or (key == "milestone" and mode == SubmissionMode.MILESTONE):
                            run_score = score
                            
                        run_id = db.create_submission_run(
                            submission_id,
                            value.start,
                            value.end,
                            mode=key,
                            runner=gpu_type.name,
                            score=run_score,
                            secret=mode == SubmissionMode.PRIVATE,
                            compilation=value.compilation,
                            result=value.run,
                            system=result.system,
                        )
                        
                        # If this is a milestone submission, record the milestone run
                        if mode == SubmissionMode.MILESTONE and run_id:
                            # Get submission data to find the leaderboard
                            submission_data = db.get_submission_by_id(submission_id)
                            if submission_data:
                                leaderboard = db.get_leaderboard(submission_data["leaderboard_name"])
                                if leaderboard:
                                    # Find the milestone ID based on the filename
                                    milestones = db.get_leaderboard_milestones(leaderboard["id"])
                                    milestone = next((m for m in milestones if m["filename"] == name), None)
                                    if milestone:
                                        db.record_milestone_run(milestone["id"], submission_id, run_id)

        return result

    @with_error_handling
    async def run_script(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        """
        Function invoked by the `run` command to run a single script.
        """
        reporter = MultiProgressReporter(interaction, "Script run")
        rep = reporter.add_run(f"{gpu_type.name}")
        await reporter.show()
        gpu_type = get_gpu_by_name(gpu_type.name)
        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return

        await self._handle_submission(
            gpu_type,
            rep,
            code=script_content,
            name=script.filename,
            task=None,
            mode=SubmissionMode.SCRIPT,
        )

    async def _handle_submission(
        self,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        code: str,
        name: str,
        task: Optional[LeaderboardTask],
        mode: SubmissionMode,
        submission_id: int = -1,
    ) -> Optional[FullResult]:
        """
        Generic function to handle code submissions.
        Args:
            interaction: Interaction that started this command.
            gpu_type: Which GPU to run on.
            code: Submitted code
            name: File name of the submission; used to infer code's language
            task: Task specification, of provided
            submission_id: ID of the submission, only used for display purposes

        Returns:
            if successful, returns the result of the run.
        """
        launcher = self.launcher_map[gpu_type.value]
        config = build_task_config(
            task=task, submission_content=code, arch=self._get_arch(gpu_type), mode=mode
        )

        logger.info("submitting task to runner %s", launcher.name)

        result = await launcher.run_submission(config, gpu_type, reporter)

        if not result.success:
            await reporter.update_title(reporter.title + " ❌ failure")
            await reporter.push(result.error)
            return result
        else:
            await reporter.update_title(reporter.title + " ✅ success")

        short_report = make_short_report(
            result.runs, full=mode in [SubmissionMode.PRIVATE, SubmissionMode.LEADERBOARD]
        )
        await reporter.push(short_report)
        if mode != SubmissionMode.PRIVATE:
            try:
                # does the last message of the short report start with ✅ or ❌?
                verdict = short_report[-1][0]
                id_str = f"{verdict}" if submission_id == -1 else f"{verdict} #{submission_id}"
                await reporter.display_report(
                    f"{id_str} {name} on {gpu_type.name} ({launcher.name})",
                    generate_report(result),
                )
            except Exception as E:
                logger.error("Error generating report. Result: %s", result, exc_info=E)
                raise

        return result

    async def _validate_input_file(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
    ) -> Optional[str]:
        # load and decode
        try:
            return (await script.read()).decode("utf-8")
        except UnicodeError:
            await send_discord_message(
                interaction,
                f"Could not decode your file `{script.filename}`.\nIs it UTF-8?",
                ephemeral=True,
            )
            return None

    def _get_arch(self, gpu_type: GPU):
        return GPU_TO_SM[gpu_type.name]
