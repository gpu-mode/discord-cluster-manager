import asyncio
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import List, Optional

import discord
from consts import (
    AllGPU,
    GitHubGPU,
    ModalGPU,
)
from discord import app_commands
from discord.ext import commands
from leaderboard_db import leaderboard_name_autocomplete
from leaderboard_eval import cu_eval, py_eval
from ui.misc import DeleteConfirmationModal, GPUSelectionView
from ui.table import create_table
from utils import (
    extract_score,
    get_user_from_id,
    send_discord_message,
    setup_logging,
)

logger = setup_logging()


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(self, bot):
        super().__init__(name="submit", description="Submit to leaderboard")
        self.bot = bot

    async def async_submit_cog_job(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        command,
        reference_code,
        submission_content,
        cog: commands.Cog,
        gpu: AllGPU,
        runner_name: str = "GitHub",
    ):
        try:
            discord_thread = await command.callback(
                cog,
                interaction,
                script,
                app_commands.Choice(
                    name=gpu.name,
                    value=gpu.value,
                ),
                reference_code=reference_code,
            )
        except discord.errors.NotFound as e:
            print(f"Webhook not found: {e}")
            await send_discord_message(interaction, "❌ The webhook was not found.")

        message_contents = [msg.content async for msg in discord_thread.history(limit=None)]

        try:
            # For CUDA leaderboards, make more robust
            if "check_implementation failed" in message_contents:
                await send_discord_message(
                    interaction,
                    "check_implementation failed. User kernel and reference kernel do not match.",
                    ephemeral=True,
                )
                return

            # TODO: Make this more robust later
            score = extract_score("".join(message_contents))

            with self.bot.leaderboard_db as db:
                db.create_submission(
                    {
                        "submission_name": script.filename,
                        "submission_time": datetime.now(),
                        "leaderboard_name": leaderboard_name,
                        "code": submission_content,
                        "user_id": interaction.user.id,
                        "submission_score": score,
                        "gpu_type": gpu.name,
                    }
                )

            user_id = (
                interaction.user.global_name
                if interaction.user.nick is None
                else interaction.user.nick
            )

            await send_discord_message(
                interaction,
                f"Successfully ran on {gpu.name} using {runner_name} runners!\n"
                + f"Leaderboard '{leaderboard_name}'.\n"
                + f"Submission title: {script.filename}.\n"
                + f"Submission user: {user_id}.\n"
                + f"Runtime: {score:.9f} seconds.",
                ephemeral=True,
            )
        except Exception:
            await send_discord_message(
                interaction,
                f"Leaderboard submission to '{leaderboard_name}' on {gpu.name} "
                + f"using {runner_name} runners failed!\n",
                ephemeral=True,
            )

    async def select_gpu_view(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        gpus: List[str],
    ):
        """
        UI displayed to user to select GPUs that they want to use.
        """
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        view = GPUSelectionView(gpus)

        await send_discord_message(
            interaction,
            f"Please select the GPU(s) for leaderboard: {leaderboard_name}.",
            view=view,
            ephemeral=True,
        )

        await view.wait()
        return view

    async def before_submit_hook(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        """
        Main logic to handle at the beginning of a user submission to a runner, to make
        sure reference code, deadlines, etc. are all correct.
        """
        # Read and convert reference code
        reference_code = None
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)

            now = datetime.now()
            deadline = leaderboard_item["deadline"]

            if now.date() > deadline.date():
                await send_discord_message(
                    interaction,
                    f"The deadline to submit to {leaderboard_name} has passed.\n"
                    + f"It was {deadline.date()} and today is {now.date()}.",
                )
                return None

            if not leaderboard_item:
                await send_discord_message(
                    interaction,
                    f"Leaderboard {leaderboard_name} not found.",
                    ephemeral=True,
                )
                return None

            leaderboard_item = db.get_leaderboard(leaderboard_name)
            gpus = db.get_leaderboard_gpu_types(leaderboard_name)
            reference_code = leaderboard_item["reference_code"]

        # Read the template file
        submission_content = await script.read()

        try:
            submission_content = submission_content.decode()
        except UnicodeError:
            await send_discord_message(
                interaction, "Could not decode your file. Is it UTF-8?", ephemeral=True
            )
            return None

        return (submission_content, reference_code, gpus)

    async def on_submit_hook(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        command,
        cog: commands.Cog,
        GPUsEnum: Enum,
        runner_name: str,
    ) -> int:
        """
        Called as the main body of a submission to route to the correct runner.
        """
        try:
            submission_content, reference_code, gpus = await self.before_submit_hook(
                interaction,
                leaderboard_name,
                script,
            )
        except Exception:
            return -1

        # GPU selection View
        gpu_enums = {e.name for e in GPUsEnum}
        gpus = [gpu for gpu in gpus if gpu in gpu_enums]

        if len(gpus) == 0:
            await send_discord_message(
                interaction,
                "❌ No available GPUs for Leaderboard "
                + f"`{leaderboard_name}` on {runner_name} runner.",
            )
            return -1

        view = await self.select_gpu_view(interaction, leaderboard_name, gpus)

        tasks = [
            self.async_submit_cog_job(
                interaction,
                leaderboard_name,
                script,
                command,
                reference_code,
                submission_content,
                cog,
                AllGPU[gpu],
                runner_name,
            )
            for gpu in view.selected_gpus
        ]

        await asyncio.gather(*tasks)
        return 0

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.channel_id != self.bot.leaderboard_submissions_id:
            await interaction.response.send_message(
                f"Please use submission commands in <#{self.bot.leaderboard_submissions_id}>",
                ephemeral=True,
            )
            return False
        return True

    # Parent command that defines global options
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    # TODO: Modularize this so all the write functionality is in here. Haven't figured
    # a good way to do this yet.
    async def submit(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        pass

    @app_commands.command(name="modal", description="Submit leaderboard data for modal")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_modal(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        # Call Modal runner
        modal_cog = self.bot.get_cog("ModalCog")

        if not all([modal_cog]):
            await send_discord_message(interaction, "❌ Required Modal cogs not found!")
            return
        modal_runner_command = modal_cog.run_modal

        success = await self.on_submit_hook(
            interaction,
            leaderboard_name,
            script,
            modal_runner_command,
            modal_cog,
            ModalGPU,
            "Modal",
        )
        return success

    @app_commands.command(name="github", description="Submit leaderboard data for GitHub")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_github(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        # Call GH runner
        github_cog = self.bot.get_cog("GitHubCog")

        if not all([github_cog]):
            await send_discord_message(interaction, "❌ Required cogs not found!")
            return
        gh_runner_command = github_cog.run_github

        success = await self.on_submit_hook(
            interaction,
            leaderboard_name,
            script,
            gh_runner_command,
            github_cog,
            GitHubGPU,
            "Modal",
        )
        return success


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot: commands.Bot = bot

        bot.leaderboard_group.add_command(LeaderboardSubmitCog(bot))

        self.get_leaderboards = bot.leaderboard_group.command(
            name="list", description="Get all leaderboards"
        )(self.get_leaderboards)

        self.leaderboard_create = bot.leaderboard_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

        self.delete_leaderboard = bot.leaderboard_group.command(
            name="delete", description="Delete a leaderboard"
        )(self.delete_leaderboard)

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="show", description="Get all submissions for a leaderboard"
        )(self.get_leaderboard_submissions)

        self.get_user_leaderboard_submissions = bot.leaderboard_group.command(
            name="show-personal", description="Get all your submissions for a leaderboard"
        )(self.get_user_leaderboard_submissions)

        self.get_leaderboard_references = bot.leaderboard_group.command(
            name="reference-code", description="Get leaderboard reference codes"
        )(self.get_leaderboard_references)

        self.get_leaderboard_eval = bot.leaderboard_group.command(
            name="eval-code", description="Get leaderboard evaluation codes"
        )(self.get_leaderboard_eval)

    # --------------------------------------------------------------------------
    # |                           HELPER FUNCTIONS                              |
    # --------------------------------------------------------------------------

    async def admin_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.get_role(self.bot.leaderboard_admin_role_id):
            return False
        return True

    async def creator_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.get_role(self.bot.leaderboard_creator_role_id):
            return True
        return False

    async def is_creator_check(
        self, interaction: discord.Interaction, leaderboard_name: str
    ) -> bool:
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if leaderboard_item["creator_id"] == interaction.user.id:
                return True
            return False

    async def _display_lb_submissions_helper(
        self,
        submissions,
        interaction,
        leaderboard_name: str,
        gpu: str,
        user_id: Optional[int] = None,
    ):
        """
        Display leaderboard submissions for a particular GPU to discord.
        Must be used as a follow-up currently.
        """

        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if not submissions:
            await send_discord_message(
                interaction,
                f'No submissions found for "{leaderboard_name}".',
                ephemeral=True,
            )
            return

        # Create embed
        processed_submissions = [
            {
                "Rank": submission["rank"],
                "User": await get_user_from_id(submission["user_id"], interaction, self.bot),
                "Score": f"{submission['submission_score']:.9f}",
                "Submission Name": submission["submission_name"],
            }
            for submission in submissions
        ]

        title = f'Leaderboard Submissions for "{leaderboard_name}" on {gpu}'
        if user_id:
            title += f" for user {await get_user_from_id(user_id, interaction, self.bot)}"

        column_widths = {
            "Rank": 4,
            "User": 14,
            "Score": 12,
            "Submission Name": 14,
        }
        embed, view = create_table(
            title,
            processed_submissions,
            items_per_page=5,
            column_widths=column_widths,
        )

        await send_discord_message(
            interaction,
            "",
            embed=embed,
            view=view,
            ephemeral=True,
        )

    async def _get_submissions_helper(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        user_id: str = None,
    ):
        """Helper method to get leaderboard submissions with optional user filtering"""
        try:
            submissions = {}
            with self.bot.leaderboard_db as db:
                leaderboard_id = db.get_leaderboard(leaderboard_name)["id"]
                if not leaderboard_id:
                    await send_discord_message(
                        interaction,
                        f'Leaderboard "{leaderboard_name}" not found.',
                        ephemeral=True,
                    )
                    return

                gpus = db.get_leaderboard_gpu_types(leaderboard_name)
                for gpu in gpus:
                    submissions[gpu] = db.get_leaderboard_submissions(
                        leaderboard_name, gpu, user_id
                    )

            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)

            view = GPUSelectionView(gpus)
            await send_discord_message(
                interaction,
                f"Please select GPUs view for leaderboard: {leaderboard_name}.",
                view=view,
                ephemeral=True,
            )

            await view.wait()

            for gpu in view.selected_gpus:
                await self._display_lb_submissions_helper(
                    submissions[gpu],
                    interaction,
                    leaderboard_name,
                    gpu,
                    user_id,
                )

        except Exception as e:
            logger.error(str(e))
            if "'NoneType' object is not subscriptable" in str(e):
                await send_discord_message(
                    interaction,
                    f"The leaderboard '{leaderboard_name}' doesn't exist.",
                    ephemeral=True,
                )
            else:
                await send_discord_message(
                    interaction, "An unknown error occurred.", ephemeral=True
                )

    # --------------------------------------------------------------------------
    # |                           COMMANDS                                      |
    # --------------------------------------------------------------------------

    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()

        if not leaderboards:
            await send_discord_message(interaction, "No leaderboards found.", ephemeral=True)
            return

        to_show = [
            {
                "Name": x["name"],
                "Deadline": x["deadline"].strftime("%Y-%m-%d %H:%M"),
                "GPU Types": ", ".join(x["gpu_types"]),
            }
            for x in leaderboards
        ]

        column_widths = {
            "Name": 18,
            "Deadline": 18,
            "GPU Types": 11,
        }
        embed, view = create_table(
            "Active Leaderboards",
            to_show,
            items_per_page=5,
            column_widths=column_widths,
        )

        await send_discord_message(
            interaction,
            "",
            embed=embed,
            view=view,
            ephemeral=True,
        )

    @app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_leaderboard_references(
        self, interaction: discord.Interaction, leaderboard_name: str
    ):
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if not leaderboard_item:
                await send_discord_message(interaction, "Leaderboard not found.", ephemeral=True)
                return

        code = leaderboard_item["reference_code"]
        code_file = StringIO(code)
        language = "cuda" if "#include" in code else "python"
        suffix = "py" if language == "python" else "cuh"

        ref_code = discord.File(
            fp=code_file, filename=f"{leaderboard_name}_reference_code.{suffix}"
        )

        message = (
            f"**Reference Code for {leaderboard_name} in {language}**\n"
            f"*If you want to display the evaluation code, run `/leaderboard eval-code {language}`*"
        )

        await send_discord_message(interaction, message, ephemeral=True, file=ref_code)

    @app_commands.describe(language="Language of the evaluation code [cpp, python]")
    @app_commands.choices(
        language=[app_commands.Choice(name=lang, value=lang) for lang in ["cuda", "python"]]
    )
    async def get_leaderboard_eval(self, interaction: discord.Interaction, language: str):
        await interaction.response.defer(ephemeral=True)

        if language == "cuda":
            eval_code = cu_eval
        else:
            eval_code = py_eval

        suffix = "py" if language == "python" else "cu"

        code_file = StringIO(eval_code)
        ref_code = discord.File(fp=code_file, filename=f"leaderboard_eval.{suffix}")

        await send_discord_message(
            interaction,
            f"**Evaluation Code for language: {language}**\n",
            file=ref_code,
            ephemeral=True,
        )

    @discord.app_commands.describe(
        leaderboard_name="Name of the leaderboard",
        deadline="Competition deadline in the form: 'Y-m-d'",
        reference_code="Reference implementation of kernel. Also includes eval code.",
    )
    async def leaderboard_create(  # noqa: C901
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        deadline: str,
        reference_code: discord.Attachment,
    ):
        is_admin = await self.admin_check(interaction)
        is_creator = await self.creator_check(interaction)
        thread = None
        if len(leaderboard_name) > 95:
            await send_discord_message(
                interaction,
                "Leaderboard name is too long. Please keep it under 95 characters.",
                ephemeral=True,
            )
            return

        if not (is_admin or is_creator):
            await send_discord_message(
                interaction,
                "You need the Leaderboard Creator role or the Leaderboard Admin role to use this command.",  # noqa: E501
                ephemeral=True,
            )
            return

        # Try parsing with time first
        # Try parsing with time first
        try:
            date_value = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                date_value = datetime.strptime(deadline, "%Y-%m-%d")
            except ValueError as ve:
                logger.error(f"Value Error: {str(ve)}", exc_info=True)
                await send_discord_message(
                    interaction,
                    "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                    ephemeral=True,
                )
                return

        # Ask the user to select GPUs
        view = GPUSelectionView([gpu.name for gpu in GitHubGPU] + [gpu.name for gpu in ModalGPU])

        await send_discord_message(
            interaction,
            "Please select GPUs for this leaderboard.",
            view=view,
            ephemeral=True,
        )

        await view.wait()

        try:
            # Read the template file
            template_content = await reference_code.read()

            with self.bot.leaderboard_db as db:
                err = db.create_leaderboard(
                    {
                        "name": leaderboard_name,
                        "deadline": date_value,
                        "reference_code": template_content.decode("utf-8"),
                        "gpu_types": view.selected_gpus,
                        "creator_id": interaction.user.id,
                    }
                )

                if err:
                    if "duplicate key" in err:
                        await send_discord_message(
                            interaction,
                            "Error: Tried to create a leaderboard "
                            f'"{leaderboard_name}" that already exists.',
                            ephemeral=True,
                        )
                    else:
                        # Handle any other errors
                        logger.error(f"Error in leaderboard creation: {err}")
                        await send_discord_message(
                            interaction,
                            "Error in leaderboard creation.",
                            ephemeral=True,
                        )
                    return

            forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)

            existing_threads = [
                thread for thread in forum_channel.threads if thread.name == leaderboard_name
            ]

            if not existing_threads:
                thread = await forum_channel.create_thread(
                    name=leaderboard_name,
                    content=(
                        f"# New Leaderboard: {leaderboard_name}\n\n"
                        f"**Deadline**: {date_value.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"**Reference Code**: {reference_code}\n\n"
                        "Submit your entries using `/submit github` or `/submit modal` in the submissions channel.\n\n"  # noqa: E501
                        f"Good luck to all participants! 🚀 <@&{self.bot.leaderboard_participant_role_id}>"  # noqa: E501
                    ),
                    auto_archive_duration=10080,  # 7 days
                )

            await send_discord_message(
                interaction,
                f"Leaderboard '{leaderboard_name}'.\n"
                + f"Reference code: {reference_code}. Submission deadline: {date_value}"
                + f"\nForum thread: {thread.thread.mention}",
            )
            return

        except discord.Forbidden:
            await send_discord_message(
                interaction,
                "Error: Bot doesn't have permission to create forum threads. Leaderboard was not created.",  # noqa: E501
                ephemeral=True,
            )
        except discord.HTTPException:
            await send_discord_message(
                interaction,
                "Error creating forum thread. Leaderboard was not created.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Error in leaderboard creation: {e}")
            # Handle any other errors
            await send_discord_message(
                interaction,
                "Error in leaderboard creation.",
                ephemeral=True,
            )
        if thread:
            await thread.thread.delete()

        with self.bot.leaderboard_db as db:  # Cleanup in case lb was created
            db.delete_leaderboard(leaderboard_name)

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name)

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_user_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name, str(interaction.user.id))

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @discord.app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def delete_leaderboard(self, interaction: discord.Interaction, leaderboard_name: str):
        is_admin = await self.admin_check(interaction)
        is_creator = await self.creator_check(interaction)
        is_creator_of_leaderboard = await self.is_creator_check(interaction, leaderboard_name)

        if not (is_admin):
            if not is_creator:
                await send_discord_message(
                    interaction,
                    "You need the Leaderboard Creator role or the Leaderboard Admin role to use this command.",  # noqa: E501
                    ephemeral=True,
                )
                return
            if not is_creator_of_leaderboard:
                await send_discord_message(
                    interaction,
                    "You need to be the creator of the leaderboard to use this command.",
                    ephemeral=True,
                )
                return

        modal = DeleteConfirmationModal("leaderboard", leaderboard_name, self.bot.leaderboard_db)
        await interaction.response.send_modal(modal)
