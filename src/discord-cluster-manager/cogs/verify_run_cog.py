import discord
from discord import app_commands
from discord.ext import commands
import re
from utils import setup_logging

logger = setup_logging()

class VerifyRunCog(commands.Cog):
    """
    A Discord cog for verifying the success of trainingruns.

    This cog provides functionality to verify that either a GitHub Actions or
    Modal run completed successfully by checking for specific message patterns
    in a Discord thread. It supports verification of two types of runs:
    1. GitHub Actions runs - Identified by "GitHub Action triggered!" message
    2. Modal runs - Identified by "Running on Modal..." message

    Commands:
        /verifyrun: Verifies the success of a run in the current thread. Can
            only be used in a thread. Automatically detects the run type and
            applies appropriate verification.
    """

    def __init__(self, bot):
        self.bot = bot

    async def verify_github_run(self, interaction: discord.Interaction, message_contents: list[str]):
        """Verify that a GitHub Actions run completed successfully"""

        required_patterns = [
            "Processing `.*` with",
            "GitHub Action triggered! Run ID:",
            "Training completed with status: success",
            ".*```\nLogs.*:",
            "View the full run at:",
        ]

        all_patterns_found = all(
            any(
                re.match(pattern, content, re.DOTALL) != None
                for content in message_contents
            )
            for pattern in required_patterns
        )

        if all_patterns_found:
            await interaction.response.send_message(
                "✅ All expected messages found - run completed successfully!")
        else:
            missing_patterns = [
                pattern for pattern in required_patterns
                if not any(re.match(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await interaction.response.send_message(
                "❌ Run verification failed. Missing expected messages:\n" +
                "\n".join(f"- {pattern}" for pattern in missing_patterns)
            )

    async def verify_modal_run(self, interaction: discord.Interaction, message_contents: list[str]):
        """Verify that a Modal run completed successfully"""

        required_patterns = [
            "Processing `.*` with",
            "Running on Modal...",
            ".*```\nModal execution result:",
        ]

        all_patterns_found = all(
            any(re.match(pattern, content, re.DOTALL) != None for content in message_contents)
            for pattern in required_patterns
        )

        if all_patterns_found:
            await interaction.response.send_message("✅ All expected messages found - Modal run completed successfully!")
        else:
            missing_patterns = [
                pattern for pattern in required_patterns
                if not any(re.match(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await interaction.response.send_message(
                "❌ Modal run verification failed. Missing expected messages:\n" +
                "\n".join(f"- {pattern}" for pattern in missing_patterns)
            )

    @app_commands.command(name='verifyrun')
    async def verify_run(self, interaction: discord.Interaction):
        """Verify that a run in the current thread completed successfully"""

        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("This command can only be used in a thread!")
            return

        message_contents = [msg.content async for msg in interaction.channel.history(limit=None)]

        # Check for GitHub Action run
        if any("GitHub Action triggered!" in content for content in message_contents):
            await self.verify_github_run(interaction, message_contents)
        # Check for Modal run
        elif any("Running on Modal..." in content for content in message_contents):
            await self.verify_modal_run(interaction, message_contents)
        else:
            await interaction.response.send_message("❌ Could not determine run type!")