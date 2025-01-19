from dataclasses import asdict

from cogs.submit_cog import SubmitCog
from consts import GPU_SELECTION
from discord import app_commands
from fastapi import FastAPI, HTTPException, UploadFile
from utils import build_task_config

app = FastAPI()

bot_instance = None


def init_api(_bot_instance):
    global bot_instance
    bot_instance = _bot_instance


class MockProgressReporter:
    """Class that pretends to be a progress reporter,
    is used to avoid errors when running submission,
    because runners report progress via discord interactions
    """

    async def push(self, message: str):
        pass

    async def update(self, message: str):
        pass


@app.post("/{runner_name}/{leaderboard_name}/{gpu_type}")
async def webhook(runner_name: str, leaderboard_name: str, gpu_type: str, file: UploadFile):
    """
    Webhook endpoint that accepts file submissions and runs them using Modal
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    runner_name = runner_name.lower()
    cog_name = {"github": "GitHubCog", "modal": "ModalCog"}[runner_name]

    gpu_name = gpu_type.upper()

    with bot_instance.leaderboard_db as db:
        leaderboard_item = db.get_leaderboard(leaderboard_name)
        reference_code = leaderboard_item["reference_code"]

    runner_cog: SubmitCog = bot_instance.get_cog(cog_name)
    config = build_task_config(
        "cu",
        reference_code,
        file.file.read().decode("utf-8"),
        runner_cog._get_arch(app_commands.Choice(name=gpu_name, value=gpu_name)),
    )

    gpu = GPU_SELECTION[runner_name.capitalize()][gpu_name]

    result = await runner_cog._run_submission(config, gpu, MockProgressReporter())

    return {"status": "success", "result": asdict(result)}
