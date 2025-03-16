import json
import signal
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

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
    """HTTP version of run_pytorch_script, handling timeouts"""
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


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Turn POST requests into runs."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            config: Dict[str, Any] = data.get("config", {})
            timeout: int = data.get("timeout", 300)

            result = local_run_config(config, timeout)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result.__dict__).encode("utf-8"))

        except Exception as e:
            error_response = {
                "success": False,
                "error": f"Server Error: {str(e)}",
                "traceback": traceback.format_exc(),
            }
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode("utf-8"))


def run_server(host="0.0.0.0", port=33001):
    """Start the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serving on {host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
