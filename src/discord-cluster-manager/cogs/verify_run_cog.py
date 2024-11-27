import discord
from discord import app_commands
from discord.ext import commands
import re
from utils import setup_logging
from unittest.mock import AsyncMock

logger = setup_logging()

def create_mock_attachment(filename: str, content: str):
    "Create an AsyncMock to simulate discord.Attachment"

    mock_attachment = AsyncMock(spec=discord.Attachment)
    mock_attachment.filename = filename
    mock_attachment.content_type = 'text/plain'
    # Simulate the read method
    mock_attachment.read = AsyncMock(return_value=content.encode('utf-8'))
    return mock_attachment

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
    
    @app_commands.command(name='verifyrun2')
    async def verify_run2(self, interaction: discord.Interaction):
        """Verify runs on on Modal, GitHub Nvidia, and GitHub AMD."""

        try:
            # Get instances of the other cogs
            modal_cog = self.bot.get_cog('ModalCog')
            github_cog = self.bot.get_cog('GitHubCog')

            if not all([modal_cog, github_cog]):
                await interaction.followup.send("❌ Required cogs not found!")
                return

            script_content = "print('Hello, world!')"
            script_file = create_mock_attachment("test_script.py", script_content)

            t4 = app_commands.Choice(name="NVIDIA T4", value="t4")
            nvidia = app_commands.Choice(name="NVIDIA", value="nvidia")
            amd = app_commands.Choice(name="AMD", value="amd")

            modal_command = modal_cog.run_modal
            await modal_command.callback(modal_cog, interaction, script_file, t4, use_followup=True)

            github_command = github_cog.run_github
            await github_command.callback(github_cog, interaction, script_file, nvidia, use_followup=True)
            await github_command.callback(github_cog, interaction, script_file, amd, use_followup=True)

            await interaction.followup.send(
               "✅ Started all verification runs:\n"
               "- Modal run\n"
               "- GitHub Nvidia run\n"
               "- GitHub AMD run"
            )

        except Exception as e:
            logger.error(f"Error starting verification runs: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ Error starting verification runs: {str(e)}"
            )