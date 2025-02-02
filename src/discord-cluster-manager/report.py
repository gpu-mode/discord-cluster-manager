import pprint
from typing import Optional

import discord

from consts import SubmissionMode
from run_eval import FullResult, CompileResult, RunResult


def _limit_length(text: str, maxlen: int):
    if len(text) > maxlen:
        return text[: maxlen - 6] + " [...]"
    else:
        return text


async def _send_split_log(thread: discord.Thread, partial_message: str, header: str, log: str):
    if len(partial_message) + len(log) + len(header) < 1900:
        partial_message += f"\n\n## {header}:\n"
        partial_message += f"```\n{log}```"
        return partial_message
    else:
        # send previous chunk
        await thread.send(partial_message)
        lines = log.splitlines()
        chunks = []
        partial_message = ""
        for line in lines:
            if len(partial_message) + len(line) < 1900:
                partial_message += line + "\n"
            else:
                chunks.append(partial_message)
                partial_message = line

        if partial_message != "":
            chunks.append(partial_message)

        # now, format the chunks
        for i, chunk in enumerate(chunks):
            partial_message = f"\n\n## {header} ({i+1}/{len(chunks)}):\n"
            partial_message += f"```\n{_limit_length(chunk, 1900)}```"
            await thread.send(partial_message)

        return ""


async def _generate_compile_report(thread: discord.Thread, comp: CompileResult):
    message = ""
    if not comp.nvcc_found:
        message += "# Compilation failed\nNVCC could not be found.\n"
        message += "This indicates a bug in the runner configuration, _not in your code_.\n"
        message += "Please notify the server admins of this problem"
        await thread.send(message)
        return

    # ok, we found nvcc
    message += "# Compilation failed\n"
    message += "Command "
    message += f"```bash\n>{_limit_length(comp.command, 1000)}```\n"
    message += f"exited with code **{comp.exit_code}**."

    message = await _send_split_log(thread, message, "Compiler stderr", comp.stderr.strip())

    if len(comp.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Compiler stdout", comp.stdout.strip())

    if len(message) != 0:
        await thread.send(message)


async def _generate_crash_report(thread: discord.Thread, run: RunResult):
    message = "# Running failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    message += f"exited with error code **{run.exit_code}** after {run.duration:.2} seconds."

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

    if len(message) != 0:
        await thread.send(message)


def format_time(value: float|str, err: Optional[float|str] = None, scale=None):
    # really ugly, but works for now
    value = float(value)

    scale = 1 # nanoseconds
    unit = "ns"
    if value > 2000:
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


async def _generate_test_report(thread: discord.Thread, run: RunResult):
    message = "# Testing failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    message += (
        f"ran successfully in {run.duration:.2} seconds, but did not pass all tests.\n"
    )

    # Generate a test
    test_log = []
    for i in range(len(run.result)):
        status = run.result.get(f"test.{i}.status", None)
        spec = run.result.get(f"test.{i}.spec", "<Error>")
        if status is None:
            break
        if status == "pass":
            test_log.append(f"✅ {spec}")
        elif status == "fail":
            test_log.append(f"❌ {spec}")
            error = run.result.get(f"test.{i}.error", "No error information available")
            if error:
                test_log.append(f"> {error}")

    if len(test_log) > 0:
        message = await _send_split_log(
            thread,
            message,
            "Test log", str.join("\n", test_log),
        )
    else:
        message += "❗ Could not find any test cases\n"

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(
            thread, message, "Program stderr", run.stderr.strip()
        )

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(
            thread, message, "Program stdout", run.stdout.strip()
        )

    if len(message) != 0:
        await thread.send(message)
    return


async def generate_report(thread: discord.Thread, result: FullResult, mode: SubmissionMode):  # noqa: C901
    if not result.success:
        message = "# Failure\n"
        message += result.error
        await thread.send(message)
        return

    comp = result.compile
    run = result.run

    # minimal error messages for private run
    if mode == SubmissionMode.PRIVATE:
        if comp is not None and not comp.success:
            await thread.send("❌ Compilation failed")
            return

        if not run.success:
            await thread.send("❌ Running failed")
            return

        if not run.passed:
            await thread.send("❌ Testing failed")
            return

        else:
            await thread.send("✅ Secret run successful")
            return

    message = ""
    if comp is not None and not comp.success:
        await _generate_compile_report(thread, comp)
        return

    if not run.success:
        await _generate_crash_report(thread, run)
        return

    if mode in [SubmissionMode.TEST, SubmissionMode.BENCHMARK, SubmissionMode.LEADERBOARD]:
        if not run.passed:
            await _generate_test_report(thread, run)
            return
        else:
            message += "✅ Testing passed"

    if mode in [SubmissionMode.BENCHMARK, SubmissionMode.BENCHMARK]:
        bench_log = []

        def log_one(base_name):
            runs = run.result.get(f"{base_name}.runs", None)
            mean = run.result.get(f"{base_name}.mean", "<Error>")
            err = run.result.get(f"{base_name}.err", None)
            best = run.result.get(f"{base_name}.best", None)
            worst = run.result.get(f"{base_name}.worst", None)
            spec = run.result.get(f"{base_name}.spec", "<Error>")
            if runs is None:
                False

            bench_log.append(f"{spec}")
            bench_log.append(f" ⏱ {format_time(mean, err)}")
            if best is not None and worst is not None:
                bench_log.append(f" ⚡ {format_time(best)} 🐌 {format_time(worst)}")

            return True

        if mode == SubmissionMode.BENCHMARK:
            for i in range(len(run.result)):
                if not log_one(f"duration.{i}"):
                    break
                bench_log.append("")

            if len(bench_log) > 0:
                message = await _send_split_log(
                    thread,
                    message,
                    "Benchmarks", str.join("\n", bench_log),
                )
            else:
                message += "❗ Could not find any benchmarks\n"
        else:
            if not log_one(f"duration"):
                message += "❗ Could not find benchmark info\n"
            else:
                message = await _send_split_log(
                    thread,
                    message,
                    "Benchmark", str.join("\n", bench_log),
                )

    if mode == SubmissionMode.SCRIPT:
        # OK, we were successful
        message += "# Success!\n"
        message += "Command "
        message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
        message += f"ran successfully in {run.duration:.2} seconds.\n"

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

    if len(message) != 0:
        await thread.send(message)
