import asyncio

import modal

from libkernelbot.consts import GPU, ModalGPU
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult, SystemInfo
from libkernelbot.utils import setup_logging

from .launcher import Launcher

logger = setup_logging(__name__)


class ModalLauncher(Launcher):
    def __init__(self, add_include_dirs: list):
        super().__init__("Modal", gpus=ModalGPU)
        self.additional_include_dirs = add_include_dirs

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        loop = asyncio.get_event_loop()
        if config["lang"] == "cu":
            config["include_dirs"] = config.get("include_dirs", []) + self.additional_include_dirs
        func_type = "pytorch" if config["lang"] == "py" else "cuda"
        func_name = f"run_{func_type}_script_{gpu_type.value.lower()}"

        logger.info(f"Starting Modal run using {func_name}")

        await status.push("⏳ Waiting for Modal run to finish...")

        result = await loop.run_in_executor(
            None,
            lambda: modal.Function.lookup("discord-bot-runner", func_name).remote(config=config),
        )

        await status.update("✅ Waiting for modal run to finish... Done")

        return result
