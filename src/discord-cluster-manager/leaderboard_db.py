from typing import Optional

import psycopg2
from consts import DATABASE_URL
from psycopg2 import Error
from utils import LeaderboardItem, SubmissionItem


class LeaderboardDB:
    def __init__(self):
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL is not defined.")
            self.connection = psycopg2.connect(DATABASE_URL, sslmode="require")
            self.cursor = self.connection.cursor()
            return True
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False

    def disconnect(self):
        """Close database connection and cursor"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def create_leaderboard(
        self,
        leaderboard: LeaderboardItem
    ) -> Optional[None]:
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.leaderboard (name, deadline, reference_code)
                VALUES (%s, %s, %s)
                """,
                (
                    leaderboard["name"],
                    leaderboard["deadline"],
                    leaderboard["reference_code"],
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            self.connection.rollback()  # Ensure rollback if error occurs
            return f"Error during leaderboard creation: {e}"
        return None

    def create_submission(self, submission: SubmissionItem):
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.submission (leaderboard_id, name,
                    user_id, code, submission_time, score)
                VALUES (
                    (SELECT id FROM leaderboard.leaderboard WHERE name = %s),
                    %s, %s, %s, %s, %s)
                """,
                (
                    submission["leaderboard_name"],
                    submission["submission_name"],
                    submission["user_id"],
                    submission["code"],
                    submission["submission_time"],
                    submission["submission_score"],
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error during leaderboard submission: {e}")
            self.connection.rollback()  # Ensure rollback if error occurs

    def get_leaderboards(self) -> list[LeaderboardItem]:
        self.cursor.execute(
            """
            SELECT id, name, deadline, reference_code
            FROM leaderboard.leaderboard
            """
        )

        return [
            LeaderboardItem(id=lb[0], name=lb[1], deadline=lb[2], reference_code=lb[3])
            for lb in self.cursor.fetchall()
        ]

    def get_leaderboard(self, leaderboard_name: str) -> int | None:
        self.cursor.execute(
            """
            SELECT id, name, deadline, reference_code
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )

        res = self.cursor.fetchone()

        if res:
            return LeaderboardItem(
                id=res[0], name=res[1], deadline=res[2], reference_code=res[3]
            )
        else:
            return None

    def get_leaderboard_submissions(
        self, leaderboard_name: str
    ) -> list[SubmissionItem]:
        self.cursor.execute(
            """
            SELECT s.name, s.user_id, s.code, s.submission_time, s.score
            FROM leaderboard.submission s
            JOIN leaderboard.leaderboard l
            ON s.leaderboard_id = l.id
            WHERE l.name = %s
            ORDER BY s.score ASC
            """,
            (leaderboard_name,),
        )

        return [
            SubmissionItem(
                leaderboard_name=leaderboard_name,
                submission_name=submission[0],
                user_id=submission[1],
                code=submission[2],
                submission_time=submission[3],
                submission_score=submission[4],
            )
            for submission in self.cursor.fetchall()
        ]


if __name__ == "__main__":
    leaderboard_db = LeaderboardDB()
    leaderboard_db.connect()
    leaderboard_db.disconnect()
