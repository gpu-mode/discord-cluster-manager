import signal
import traceback
from contextlib import contextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from run_eval import FullResult, SystemInfo, run_config


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutException after specified seconds"""

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Script execution timed out after {seconds} seconds")

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def local_run_config(  # noqa: C901
    config: dict,
    timeout_seconds: int = 300,
) -> FullResult:
    """Local version of run_pytorch_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            return run_config(config)
    except TimeoutException as e:
        return FullResult(
            success=False,
            error=f"Timeout Error: {str(e)}",
            runs={},
            system=SystemInfo(),
        )
    except Exception as e:
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(),
        )


def build_app():
    app = FastAPI()

    class ConfigRequest(BaseModel):
        config: Dict[str, Any]
        timeout: int = 300

    @app.post("/")
    async def _run_config(request: ConfigRequest):
        try:
            result = local_run_config(request.config, request.timeout)
            return result.__dict__
        except Exception as e:
            error_response = {
                "success": False,
                "error": f"Server Error: {str(e)}",
                "traceback": traceback.format_exc(),
            }
            raise HTTPException(status_code=500, detail=error_response) from e


if __name__ == "__main__":
    import uvicorn

    app = build_app()

    uvicorn.run(app, host="0.0.0.0", port=8000)
