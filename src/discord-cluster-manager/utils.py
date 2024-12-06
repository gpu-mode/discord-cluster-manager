import logging
import subprocess
import datetime
from typing import TypedDict


def setup_logging():
    """Configure and setup logging for the application"""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def get_github_branch_name():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("/", 1)[1]
    except subprocess.CalledProcessError:
        return "main"


class LeaderboardItem(TypedDict):
    name: str
    deadline: datetime.datetime
    reference_code: str


class SubmissionItem(TypedDict):
    submission_name: str
    submission_time: datetime.datetime
    submission_score: float
    leaderboard_name: str
    code: str
    user_id: int


class ProfilingItem(TypedDict):
    submission_name: str
    ncu_output: str
    stdout: str
