"""
user-info-add-cli-id
"""

from yoyo import step

__depends__ = {"20250316_01_5oMi3-remember-forum-id"}

steps = [step("ALTER TABLE leaderboard.user_info ADD COLUMN cli_id VARCHAR(255) DEFAULT NULL;")]
