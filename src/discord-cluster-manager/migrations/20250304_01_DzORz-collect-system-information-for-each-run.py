"""
Collect system information for each run.
"""

from yoyo import step

__depends__ = {"20250221_01_GA8ro-submission-collection"}

steps = [
    step(
        "ALTER TABLE leaderboard.run ADD COLUMN system_info JSONB NOT NULL DEFAULT '{}'::jsonb;",
    ),
    step("ALTER TABLE leaderboard.run ALTER COLUMN system_info DROP DEFAULT;"),
]
