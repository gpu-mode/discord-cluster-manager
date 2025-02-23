import dataclasses
import datetime
import json
import logging
from typing import List, Optional

import discord
import psycopg2
from env import (
    DATABASE_URL,
    DISABLE_SSL,
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from run_eval import CompileResult, RunResult
from task import LeaderboardTask
from utils import (
    KernelBotError,
    LeaderboardItem,
    LeaderboardRankedEntry,
    LRUCache,
    RunItem,
    SubmissionItem,
    setup_logging,
)

leaderboard_name_cache = LRUCache(max_size=512)

logger = setup_logging(__name__)


async def leaderboard_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    try:
        cached_value = leaderboard_name_cache[current]
        if cached_value is not None:
            return cached_value

        bot = interaction.client
        with bot.leaderboard_db as db:
            leaderboards = db.get_leaderboard_names()
        filtered = [lb for lb in leaderboards if current.lower() in lb.lower()]
        leaderboard_name_cache[current] = [
            discord.app_commands.Choice(name=name, value=name) for name in filtered[:25]
        ]
        return leaderboard_name_cache[current]
    except Exception as e:
        logger.exception("Error in leaderboard autocomplete", exc_info=e)
        return []


class LeaderboardDB:
    def __init__(self, host: str, database: str, user: str, password: str, port: str = "5432"):
        """Initialize database connection parameters"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            self.connection = (
                psycopg2.connect(DATABASE_URL, sslmode="require" if not DISABLE_SSL else "disable")
                if DATABASE_URL
                else psycopg2.connect(**self.connection_params)
            )
            self.cursor = self.connection.cursor()
            return True
        except psycopg2.Error as e:
            logger.exception("Error connecting to PostgreSQL", exc_info=e)
            return False

    def disconnect(self):
        """Close database connection and cursor"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.cursor = None
        self.connection = None

    def __enter__(self):
        """Context manager entry"""
        assert self.connection is None, "Nested db __enter__"
        if self.connect():
            return self
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def create_leaderboard(self, leaderboard: LeaderboardItem) -> int:
        try:
            self.cursor.execute(
                """
                INSERT INTO leaderboard.leaderboard (name, deadline, task, creator_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    leaderboard["name"],
                    leaderboard["deadline"],
                    leaderboard["task"].to_str(),
                    leaderboard["creator_id"],
                ),
            )

            leaderboard_id = self.cursor.fetchone()[0]

            if isinstance(leaderboard["gpu_types"], str):
                gpu_types = [leaderboard["gpu_types"]]
            else:
                gpu_types = leaderboard["gpu_types"]

            for gpu_type in gpu_types:
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.gpu_type (leaderboard_id, gpu_type)
                    VALUES (%s, %s)
                    """,
                    (leaderboard_id, gpu_type),
                )

            self.connection.commit()
            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
            return leaderboard_id
        except psycopg2.Error as e:
            logger.exception("Error in leaderboard creation.", e)
            if isinstance(e, psycopg2.errors.UniqueViolation):
                raise KernelBotError(
                    "Error: Tried to create a leaderboard "
                    f'"{leaderboard["name"]}" that already exists.'
                ) from e
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error in leaderboard creation.") from e

    def update_leaderboard(self, name, deadline, task):
        try:
            self.cursor.execute(
                """
                UPDATE leaderboard.leaderboard
                SET deadline = %s, task = %s
                WHERE name = %s;
                """,
                (deadline, task.to_str(), name),
            )
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Error during leaderboard update", exc_info=e)
            raise KernelBotError("Error during leaderboard update") from e

    def delete_leaderboard(self, leaderboard_name: str, force: bool = False):
        try:
            if force:
                self.cursor.execute(
                    """
                    DELETE FROM leaderboard.runs
                    WHERE submission_id IN (
                        SELECT leaderboard.submission.id
                        FROM leaderboard.submission
                        WHERE leaderboard.submission.leaderboard_id IN (
                            SELECT leaderboard.leaderboard.id FROM leaderboard.leaderboard
                                WHERE leaderboard.leaderboard.name = %s
                        )
                    );
""",
                    (leaderboard_name,),
                )
                self.cursor.execute(
                    """
                    DELETE FROM leaderboard.submission
                    USING leaderboard.leaderboard
                    WHERE leaderboard.submission.leaderboard_id = leaderboard.leaderboard.id
                        AND leaderboard.leaderboard.name = %s;
                    """,
                    (leaderboard_name,),
                )

            self.cursor.execute(
                """
                DELETE FROM leaderboard.leaderboard WHERE name = %s
                """,
                (leaderboard_name,),
            )
            self.connection.commit()
            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.exception("Could not delete leaderboard %s.", leaderboard_name, exc_info=e)
            raise KernelBotError(f"Could not delete leaderboard {leaderboard_name}.") from e

    def create_submission(
        self, leaderboard: str, file_name: str, user_id: int, code: str, time: datetime.datetime
    ) -> Optional[int]:
        try:
            # check if we already have the code
            self.cursor.execute(
                """
                SELECT id, code
                FROM leaderboard.code_files
                WHERE hash = encode(sha256(%s::bytea), 'hex')
                """,
                (code,),
            )

            code_id = None
            for candidate in self.cursor.fetchall():
                if candidate[1] == code:
                    code_id = candidate[0]
                    break

            if code_id is None:
                # a genuinely new submission
                self.cursor.execute(
                    """
                    INSERT INTO leaderboard.code_files (CODE)
                    VALUES (%s)
                    RETURNING id
                    """,
                    (code,),
                )
                code_id = self.cursor.fetchone()

            self.cursor.execute(
                """
                INSERT INTO leaderboard.submission (leaderboard_id, file_name,
                    user_id, code_id, submission_time)
                VALUES (
                    (SELECT id FROM leaderboard.leaderboard WHERE name = %s),
                    %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    leaderboard,
                    file_name,
                    user_id,
                    code_id,
                    time,
                ),
            )
            submission_id = self.cursor.fetchone()[0]
            assert submission_id is not None
            self.connection.commit()
            return submission_id
        except psycopg2.Error as e:
            logger.error(
                "Error during creation of submission for leaderboard '%s' by user '%s'",
                leaderboard,
                user_id,
                exc_info=e,
            )
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error during creation of submission") from e

    def mark_submission_done(
        self,
        submission: int,
    ) -> Optional[int]:
        try:
            self.cursor.execute(
                """
                UPDATE leaderboard.submission
                SET done = TRUE
                WHERE id = %s
                """,
                (submission,),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            logger.error("Could not mark submission '%s' as done.", submission, exc_info=e)
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Error while finalizing submission") from e

    def create_submission_run(
        self,
        submission: int,
        start: datetime.datetime,
        end: datetime.datetime,
        mode: str,
        secret: bool,
        runner: str,
        score: Optional[float],
        compilation: Optional[CompileResult],
        result: RunResult,
    ):
        try:
            if compilation is not None:
                compilation = json.dumps(dataclasses.asdict(compilation))

            meta = {
                k: result.__dict__[k]
                for k in ["stdout", "stderr", "success", "exit_code", "command", "duration"]
            }
            self.cursor.execute(
                """
                INSERT INTO leaderboard.runs (submission_id, start_time, end_time, mode,
                secret, runner, score, passed, compilation, meta, result
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    submission,
                    start,
                    end,
                    mode,
                    secret,
                    runner,
                    score,
                    result.passed,
                    compilation,
                    json.dumps(meta),
                    json.dumps(result.result),
                ),
            )
            self.connection.commit()
        except psycopg2.Error as e:
            logger.exception(
                "Error during adding %s run on %s for submission '%s'",
                mode,
                runner,
                submission,
                exc_info=e,
            )
            self.connection.rollback()  # Ensure rollback if error occurs
            raise KernelBotError("Could not create leaderboard submission entry in database") from e

    def get_leaderboard_names(self) -> list[str]:
        self.cursor.execute("SELECT name FROM leaderboard.leaderboard")
        return [x[0] for x in self.cursor.fetchall()]

    def get_leaderboards(self) -> list[LeaderboardItem]:
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id
            FROM leaderboard.leaderboard
            """
        )

        lbs = self.cursor.fetchall()
        leaderboards = []

        for lb in lbs:
            self.cursor.execute(
                "SELECT * from leaderboard.gpu_type WHERE leaderboard_id = %s", [lb[0]]
            )
            gpu_types = [x[1] for x in self.cursor.fetchall()]

            leaderboards.append(
                LeaderboardItem(
                    id=lb[0],
                    name=lb[1],
                    deadline=lb[2],
                    task=LeaderboardTask.from_dict(lb[3]),
                    gpu_types=gpu_types,
                    creator_id=lb[4],
                )
            )

        return leaderboards

    def get_leaderboard_gpu_types(self, leaderboard_name: str) -> List[str] | None:
        self.cursor.execute(
            """
            SELECT *
            FROM leaderboard.gpu_type
            WHERE leaderboard_id = (
                SELECT id
                FROM leaderboard.leaderboard
                WHERE name = %s
            )
            """,
            (leaderboard_name,),
        )

        gpu_types = [x[1] for x in self.cursor.fetchall()]

        if gpu_types:
            return gpu_types
        else:
            return None

    def get_leaderboard(self, leaderboard_name: str) -> LeaderboardItem | None:
        self.cursor.execute(
            """
            SELECT id, name, deadline, task, creator_id
            FROM leaderboard.leaderboard
            WHERE name = %s
            """,
            (leaderboard_name,),
        )

        res = self.cursor.fetchone()

        if res:
            task = LeaderboardTask.from_dict(res[3])
            return LeaderboardItem(
                id=res[0],
                name=res[1],
                deadline=res[2],
                task=task,
                creator_id=res[4],
            )
        else:
            return None

    def get_leaderboard_submissions(
        self, leaderboard_name: str, gpu_name: str, user_id: Optional[str] = None
    ) -> list[LeaderboardRankedEntry]:
        # separate cases, for personal we want all submissions, for general we want best per user
        if user_id:
            # Query all if user_id (means called from show-personal)
            query = """
                SELECT
                    s.file_name,
                    s.user_id,
                    s.submission_time,
                    r.score,
                    r.runner,
                    RANK() OVER (ORDER BY r.score ASC) as rank
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = %s
                    AND r.runner = %s
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                    AND s.user_id = %s
                ORDER BY r.score ASC
                """
            args = (leaderboard_name, gpu_name, user_id)
        else:
            # Query best submission per user if no user_id (means called from show)
            query = """
                WITH best_submissions AS (
                    SELECT DISTINCT ON (s.user_id)
                        s.file_name,
                        s.user_id,
                        s.submission_time,
                        r.score,
                        r.runner
                    FROM leaderboard.runs r
                    JOIN leaderboard.submission s ON r.submission_id = s.id
                    JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                    WHERE l.name = %s AND r.runner = %s AND NOT r.secret
                          AND r.score IS NOT NULL AND r.passed
                    ORDER BY s.user_id, r.score ASC
                )
                SELECT
                    file_name,
                    user_id,
                    submission_time,
                    score,
                    runner,
                    RANK() OVER (ORDER BY score ASC) as rank
                FROM best_submissions
                ORDER BY score ASC
                """
            args = (leaderboard_name, gpu_name)

        self.cursor.execute(query, args)

        return [
            LeaderboardRankedEntry(
                leaderboard_name=leaderboard_name,
                submission_name=submission[0],
                user_id=submission[1],
                submission_time=submission[2],
                submission_score=submission[3],
                gpu_type=gpu_name,
                rank=submission[5],
            )
            for submission in self.cursor.fetchall()
        ]

    def generate_stats(self):
        try:
            return self._generate_stats()
        except Exception as e:
            logging.exception("error generating stats", exc_info=e)
            raise

    def _generate_stats(self):
        # code-level stats
        self.cursor.execute(
            """
            SELECT COUNT(*) FROM leaderboard.code_files;
            """
        )
        num_unique_codes = self.cursor.fetchone()[0]

        # submission-level stats
        self.cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE NOT done),
                COUNT(DISTINCT user_id),
                COUNT(*) FILTER (WHERE submission_time > %s)
            FROM leaderboard.submission;
            """,
            (datetime.datetime.now() - datetime.timedelta(days=1),),
        )
        num_sub, num_sub_wait, num_users, num_last_day = self.cursor.fetchone()

        # run-level stats
        self.cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE passed),
                COUNT(score),
                COUNT(*) FILTER (WHERE secret)
            FROM leaderboard.runs;
            """
        )
        num_run, num_run_pass, num_scored, num_secret = self.cursor.fetchone()

        # per-runner stats
        self.cursor.execute(
            """
            SELECT
                runner,
                COUNT(*),
                COUNT(*) FILTER (WHERE passed),
                COUNT(score),
                COUNT(*) FILTER (WHERE secret)
            FROM leaderboard.runs
            GROUP BY runner;
            """
        )

        result = {
            "num_unique_codes": num_unique_codes,
            "num_submissions": num_sub,
            "sub_waiting": num_sub_wait,
            "num_users": num_users,
            "sub_last_day": num_last_day,
            "num_runs": num_run,
            "runs_passed": num_run_pass,
            "runs_scored": num_scored,
            "runs_secret": num_secret,
        }

        for row in self.cursor.fetchall():
            result[f"num_run.{row[0]}"] = row[1]
            result[f"runs_passed.{row[0]}"] = row[2]
            result[f"runs_scored.{row[0]}"] = row[3]
            result[f"runs_secret.{row[0]}"] = row[4]

        # calculate heavy hitters
        self.cursor.execute(
            """
            WITH run_durations AS (
                SELECT
                    s.user_id AS user_id,
                    r.end_time - r.start_time AS duration
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                WHERE NOW() - s.submission_time <= interval '24 hours'
            )
            SELECT
                user_id,
                SUM(duration) AS total
            FROM run_durations
            GROUP BY user_id
            ORDER BY total DESC
            LIMIT 10;
            """
        )

        for row in self.cursor.fetchall():
            result[f"total.{row[0]}"] = row[1]

        return result

    def get_submission_by_id(self, submission_id: int) -> Optional[SubmissionItem]:
        query = """
                SELECT s.leaderboard_id, lb.name, s.file_name, s.user_id,
                       s.submission_time, s.done, c.code
                FROM leaderboard.submission s
                JOIN leaderboard.code_files c ON s.code_id = c.id
                JOIN leaderboard.leaderboard lb ON s.leaderboard_id = lb.id
                WHERE s.id = %s
                """
        self.cursor.execute(query, (submission_id,))
        submission = self.cursor.fetchone()
        if submission is None:
            return None

        # OK, now get the runs
        query = """
                SELECT start_time, end_time, mode, secret, runner, score,
                       passed, compilation, meta, result
                FROM leaderboard.runs
                WHERE submission_id = %s
                """
        self.cursor.execute(query, (submission_id,))
        runs = self.cursor.fetchall()

        runs = [
            RunItem(
                start_time=r[0],
                end_time=r[1],
                mode=r[2],
                secret=r[3],
                runner=r[4],
                score=r[5],
                passed=r[6],
                compilation=r[7],
                meta=r[8],
                result=r[9],
            )
            for r in runs
        ]

        return SubmissionItem(
            leaderboard_id=submission[0],
            leaderboard_name=submission[1],
            file_name=submission[2],
            user_id=submission[3],
            submission_time=submission[4],
            done=submission[5],
            code=submission[6],
            runs=runs,
        )


if __name__ == "__main__":
    print(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )

    leaderboard_db = LeaderboardDB(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )
    leaderboard_db.connect()
    leaderboard_db.disconnect()
