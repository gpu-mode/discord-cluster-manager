import asyncio
from typing import Optional

import discord
import modal
from consts import ModalGPU
from discord import app_commands
from discord.ext import commands
from leaderboard_eval import cu_eval, py_eval
from utils import send_discord_message, send_logs, setup_logging

logger = setup_logging()


class ModalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.run_modal = bot.run_group.command(
            name="modal", description="Run a script using Modal"
        )(self.run_modal)

    @app_commands.describe(
        script="The Python script file to run", gpu_type="Choose the GPU type for Modal"
    )
    @app_commands.choices(
        gpu_type=[app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in ModalGPU]
    )
    async def run_modal(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        reference_script: Optional[discord.Attachment] = None,
        reference_code: str = None,
    ) -> discord.Thread:
        thread = None
        status_msg = None
        try:
            if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
                await send_discord_message(
                    interaction,
                    "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
                    ephemeral=True,
                )
                return None

            # TODO: Maybe find a better way?
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)
            channel = interaction.channel
            message = await channel.send(f"Starting Modal job with {gpu_type.name}...")
            thread = await message.create_thread(name=f"{gpu_type.name} Modal Job")

            script_content = (await script.read()).decode("utf-8")
            status_msg = await thread.send(
                "**Running on Modal...**\n> ⏳ Waiting for available GPU..."
            )

            filename = "train.py" if script.filename.endswith(".py") else "train.cu"
            reference_content = None
            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )

            result, score = await self.handle_modal_execution(
                interaction,
                thread,
                script_content,
                filename,
                gpu_type.value,
                reference_content,
                status_msg,
            )

            if result is not None and score > 0:
                await thread.send(f"**score:{score:.9f}**")

            return thread

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread and status_msg:
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                await thread.send(f"**Error:** {str(e)}")
            raise

    async def handle_modal_execution(
        self,
        interaction: discord.Interaction,
        thread: discord.Thread,
        script_content: str,
        filename: str,
        gpu_type: str,
        reference_content: Optional[str],
        status_msg: discord.Message,
    ) -> tuple[str, float]:
        try:
            loop = asyncio.get_event_loop()
            func_type = "pytorch" if filename.endswith(".py") else "cuda"
            func_name = f"run_{func_type}_script_{gpu_type.lower()}"

            if reference_content is not None:
                result, score = await loop.run_in_executor(
                    None,
                    lambda: modal.Function.lookup("discord-bot-runner", func_name).remote(
                        py_eval if filename.endswith(".py") else cu_eval,
                        reference_content=reference_content,
                        submission_content=script_content,
                    ),
                )
            else:
                result, score = await loop.run_in_executor(
                    None,
                    lambda: modal.Function.lookup("discord-bot-runner", func_name).remote(
                        script_content,
                    ),
                )
                await send_discord_message(
                    interaction, f"Modal job completed in thread {thread.jump_url}", ephemeral=True
                )

            # Send results
            await thread.send(f"\n**Script size:** {len(script_content)} bytes")
            await thread.send(f"**Execution time:** {score:.3f} s\n")

            if "check_implementation failed" in result or "Error" in result:
                await thread.send("Modal run failed.\n")
                await send_logs(thread, result)
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                return result, 0

            if result is not None:
                await thread.send(f"**score:{score:.9f}**\n```")

            await status_msg.edit(content="**Running on Modal...**\n> ✅ Job completed!")
            return result, score

        except Exception as e:
            logger.error(f"Error in handle_modal_execution: {str(e)}", exc_info=True)
            await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
            await thread.send(f"**Error:** {str(e)}")
            raise
