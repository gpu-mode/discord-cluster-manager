import logging
import subprocess
from typing import Any, Optional


def setup_logging(name: Optional[str] = None):
    """Configure and setup logging for the application"""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


class KernelBotError(Exception):
    """
    This class represents an Exception that has been sanitized,
    i.e., whose message can be safely displayed to the user without
    risk of leaking internal bot details.
    """

    def __init__(self, message, code: int = 400):
        super().__init__(message)
        self.http_code = 400


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


class LRUCache:
    def __init__(self, max_size: int):
        """LRU Cache implementation, as functools.lru doesn't work in async code
        Note: Implementation uses list for convenience because cache is small, so
        runtime complexity does not matter here.
        Args:
            max_size (int): Maximum size of the cache
        """
        self._cache = {}
        self._max_size = max_size
        self._q = []

    def __getitem__(self, key: Any, default: Any = None) -> Any | None:
        if key not in self._cache:
            return default

        self._q.remove(key)
        self._q.append(key)
        return self._cache[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._q.remove(key)
            self._q.append(key)
            self._cache[key] = value
            return

        if len(self._cache) >= self._max_size:
            self._cache.pop(self._q.pop(0))

        self._cache[key] = value
        self._q.append(key)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def invalidate(self):
        """Invalidate the cache, clearing all entries, should be called when updating the underlying
        data in db
        """
        self._cache.clear()
        self._q.clear()


def format_time(value: float | str, err: Optional[float | str] = None, scale=None):  # noqa: C901
    if value is None:
        logging.warning("Expected a number, got None", stack_info=True)
        return "–"

    # really ugly, but works for now
    value = float(value)

    scale = 1  # nanoseconds
    unit = "ns"
    if value > 2_000_000:
        scale = 1000_000
        unit = "ms"
    elif value > 2000:
        scale = 1000
        unit = "µs"

    value /= scale
    if err is not None:
        err = float(err)
        err /= scale
    if value < 1:
        if err:
            return f"{value} ± {err} {unit}"
        else:
            return f"{value} {unit}"
    elif value < 10:
        if err:
            return f"{value:.2f} ± {err:.3f} {unit}"
        else:
            return f"{value:.2f} {unit}"
    elif value < 100:
        if err:
            return f"{value:.1f} ± {err:.2f} {unit}"
        else:
            return f"{value:.1f} {unit}"
    else:
        if err:
            return f"{value:.0f} ± {err:.1f} {unit}"
        else:
            return f"{value:.0f} {unit}"


def limit_length(text: str, maxlen: int):
    if len(text) > maxlen:
        return text[: maxlen - 6] + " [...]"
    else:
        return text
