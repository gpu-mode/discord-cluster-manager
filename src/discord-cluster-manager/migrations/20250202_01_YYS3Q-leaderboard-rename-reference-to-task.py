"""
leaderboard-rename-reference-to-task
"""

from yoyo import step

__depends__ = {"20250106_01_Sgph3-add-leaderboard-creator-id"}

steps = [
    step("ALTER TABLE leaderboard.leaderboard RENAME COLUMN reference_code TO task;"),
    step("ALTER TABLE leaderboard.leaderboard ALTER COLUMN task TYPE JSONB USING task::jsonb;"),
]
