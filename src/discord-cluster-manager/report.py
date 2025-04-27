import dataclasses
from typing import List

import consts
import discord
from run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo
from utils import format_time


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
        if len(partial_message) > 0:
            await thread.send(partial_message)
        lines = log.splitlines()
        chunks = []
        partial_message = ""
        for line in lines:
            if len(partial_message) + len(line) < 1900:
                partial_message += line + "\n"
            else:
                if partial_message != "":
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


@dataclasses.dataclass
class Text:
    """
    Text represents markdown-formatted text to be added to the report.
    """
    text: str


@dataclasses.dataclass
class Log:
    """
    Log represents a potentially extensive log of some operation, such as
    stdout/stderr of the compiler or tester script.
    Logs will be automatically wrapped in code blocks, and prefixed with
    the given header. If `content` is too long to fit into a single discord
    message, it can be broken up automatically (and reasonably) into multiple
    smaller messages.
    """
    header: str
    content: str


class RunResultReport:
    def __init__(self):
        self.data: List[Text | Log] = []

    def add_text(self, section: str):
        self.data.append(Text(section))

    def add_log(self, header: str, log: str):
        self.data.append(Log(header, log))


def _generate_compile_report(reporter: "RunResultReport", comp: CompileResult):
    message = ""
    if not comp.nvcc_found:
        message += "# Compilation failed\nNVCC could not be found.\n"
        message += "This indicates a bug in the runner configuration, _not in your code_.\n"
        message += "Please notify the server admins of this problem"
        reporter.add_text(message)
        return

    # ok, we found nvcc
    message += "# Compilation failed\n"
    message += "Command "
    message += f"```bash\n>{_limit_length(comp.command, 1000)}```\n"
    message += f"exited with code **{comp.exit_code}**."
    reporter.add_text(message)

    reporter.add_log("Compiler stderr", comp.stderr.strip())

    if len(comp.stdout.strip()) > 0:
        reporter.add_log("Compiler stdout", comp.stdout.strip())


def _generate_crash_report(reporter: "RunResultReport", run: RunResult):
    message = "# Running failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    if run.exit_code == consts.ExitCode.TIMEOUT_EXPIRED:
        message += f"**timed out** after {float(run.duration):.2f} seconds."
    else:
        message += (
            f"exited with error code **{run.exit_code}** after {float(run.duration):.2f} seconds."
        )
    reporter.add_text(message)

    if len(run.stderr.strip()) > 0:
        reporter.add_log("Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        reporter.add_log("Program stdout", run.stdout.strip())


def _generate_test_report(reporter: "RunResultReport", run: RunResult):
    message = "# Testing failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    message += f"ran successfully in {run.duration:.2f} seconds, but did not pass all tests.\n"
    reporter.add_text(message)

    # Generate a test
    reporter.add_log("Test log", make_test_log(run))

    if len(run.stderr.strip()) > 0:
        reporter.add_log("Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        reporter.add_log("Program stdout", run.stdout.strip())


def _short_fail_reason(run: RunResult):
    """
    Translate the exit code of `run` into a short error identifier.
    """
    if run.exit_code == consts.ExitCode.TIMEOUT_EXPIRED:
        return " (timeout)"
    elif run.exit_code == consts.ExitCode.CUDA_FAIL:
        return " (cuda api error)"
    elif run.exit_code != consts.ExitCode.VALIDATE_FAIL:
        return f" (internal error {run.exit_code})"
    else:
        return ""


def make_short_report(runs: dict[str, EvalResult], full=True) -> list[str]:  # noqa: C901
    """
    Creates a minimalistic report for `runs`,
    returned as a list of status strings
    """
    any_compile = False
    result = []
    for r in runs.values():
        if r.compilation is not None:
            any_compile = True
            if not r.compilation.success:
                return ["❌ Compilation failed"]

    if any_compile:
        result.append("✅ Compilation successful")

    if "test" in runs:
        test_run = runs["test"].run
        if not test_run.success:
            result.append("❌ Running tests failed" + _short_fail_reason(test_run))
            return result
        elif not test_run.passed:
            result.append("❌ Testing failed")
            return result
        else:
            result.append("✅ Testing successful")
    elif full:
        result.append("❌ Tests missing")

    if "benchmark" in runs:
        bench_run = runs["benchmark"].run
        if not bench_run.success:
            result.append("❌ Running benchmarks failed" + _short_fail_reason(bench_run))
            return result
        elif not bench_run.passed:
            result.append("❌ Benchmarking failed")
            return result
        else:
            result.append("✅ Benchmarking successful")
    elif full:
        result.append("❌ Benchmarks missing")

    if "leaderboard" in runs:
        lb_run = runs["leaderboard"].run
        if not lb_run.success:
            result.append("❌ Running leaderboard failed" + _short_fail_reason(lb_run))
        elif not lb_run.passed:
            result.append("❌ Leaderboard run failed")
        else:
            result.append("✅ Leaderboard run successful")
    elif full:
        result.append("❌ Leaderboard missing")
    return result


def make_test_log(run: RunResult) -> str:
    test_log = []
    for i in range(len(run.result)):
        status = run.result.get(f"test.{i}.status", None)
        spec = run.result.get(f"test.{i}.spec", "<Error>")
        if status is None:
            break
        if status == "pass":
            test_log.append(f"✅ {spec}")
            msg = run.result.get(f"test.{i}.message", None)
            if msg:
                test_log.append(f"> {msg.replace('\\n', '\n')}")
        elif status == "fail":
            test_log.append(f"❌ {spec}")
            error = run.result.get(f"test.{i}.error", "No error information available")
            if error:
                test_log.append(f"> {error.replace('\\n', '\n')}")
    if len(test_log) > 0:
        return str.join("\n", test_log)
    else:
        return "❗ Could not find any test cases"


def make_benchmark_log(run: RunResult) -> str:
    num_bench = int(run.result.get("benchmark-count", 0))

    def log_one(base_name):
        status = run.result.get(f"{base_name}.status")
        spec = run.result.get(f"{base_name}.spec")
        if status == "fail":
            bench_log.append(f"❌ {spec} failed testing:\n")
            bench_log.append(run.result.get(f"{base_name}.error"))
            return

        mean = run.result.get(f"{base_name}.mean")
        err = run.result.get(f"{base_name}.err")
        best = run.result.get(f"{base_name}.best")
        worst = run.result.get(f"{base_name}.worst")

        bench_log.append(f"{spec}")
        bench_log.append(f" ⏱ {format_time(mean, err)}")
        if best is not None and worst is not None:
            bench_log.append(f" ⚡ {format_time(best)} 🐌 {format_time(worst)}")

    bench_log = []
    for i in range(num_bench):
        log_one(f"benchmark.{i}")
        bench_log.append("")

    if len(bench_log) > 0:
        return "\n".join(bench_log)
    else:
        return "❗ Could not find any benchmarks"


def generate_system_info(system: SystemInfo):
    return f"""
Running on:
* GPU: `{system.gpu}`
* CPU: `{system.cpu}`
* Platform: `{system.platform}`
* Torch: `{system.torch}`
"""


def generate_report(result: FullResult) -> RunResultReport:  # noqa: C901
    runs = result.runs
    report = RunResultReport()
    report.add_text(generate_system_info(result.system))

    if "test" in runs:
        test_run = runs["test"]

        if test_run.compilation is not None and not test_run.compilation.success:
            _generate_compile_report(report, test_run.compilation)
            return report

        test_run = test_run.run

        if not test_run.success:
            _generate_crash_report(report, test_run)
            return report

        if not test_run.passed:
            _generate_test_report(report, test_run)
            return report
        else:
            num_tests = int(test_run.result.get("test-count", 0))
            report.add_log(f"✅ Passed {num_tests}/{num_tests} tests", make_test_log(test_run))

    if "benchmark" in runs:
        bench_run = runs["benchmark"]
        if bench_run.compilation is not None and not bench_run.compilation.success:
            _generate_compile_report(report, bench_run.compilation)
            return report

        bench_run = bench_run.run
        if not bench_run.success:
            _generate_crash_report(report, bench_run)
            return report

        report.add_log(
            "Benchmarks",
            make_benchmark_log(bench_run),
        )

    if "leaderboard" in runs:
        bench_run = runs["leaderboard"]
        if bench_run.compilation is not None and not bench_run.compilation.success:
            _generate_compile_report(report, bench_run.compilation)
            return report

        bench_run = bench_run.run
        if not bench_run.success:
            _generate_crash_report(report, bench_run)
            return report

        report.add_log(
            "Ranked Benchmark",
            make_benchmark_log(bench_run),
        )

    if "script" in runs:
        run = runs["script"]
        if run.compilation is not None and not run.compilation.success:
            _generate_compile_report(report, run.compilation)
            return report

        run = run.run
        # OK, we were successful
        message = "# Success!\n"
        message += "Command "
        message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
        message += f"ran successfully in {run.duration:.2f} seconds.\n"
        report.add_text(message)

    if len(runs) == 1:
        run = next(iter(runs.values()))
        if len(run.run.stderr.strip()) > 0:
            report.add_log("Program stderr", run.run.stderr.strip())

        if len(run.run.stdout.strip()) > 0:
            report.add_log("Program stdout", run.run.stdout.strip())

    return report


class MultiProgressReporter:
    def __init__(self, interaction: discord.Interaction, header: str):
        self.header = header
        self.runs = []
        self.interaction = interaction

    async def show(self):
        await self._update_message()

    def add_run(self, title: str) -> "RunProgressReporter":
        rpr = RunProgressReporterDiscord(self, self.interaction, title)
        self.runs.append(rpr)
        return rpr

    def make_message(self):
        formatted_runs = []
        for run in self.runs:
            formatted_runs.append(run.get_message())

        return str.join("\n\n", [f"# {self.header}"] + formatted_runs)

    async def _update_message(self):
        if self.interaction is None:
            return

        await self.interaction.edit_original_response(content=self.make_message(), view=None)


class RunProgressReporter:
    def __init__(self, title: str):
        # short report
        self.title = title
        self.lines = []

    async def push(self, content: str | list[str]):
        if isinstance(content, str):
            self.lines.append(f"> {content}")
        else:
            for line in content:
                self.lines.append(f"> {line}")
        await self._update_message()

    async def update(self, new_content: str):
        self.lines[-1] = f"> {new_content}"
        await self._update_message()

    async def update_title(self, new_title):
        self.title = new_title
        await self._update_message()

    def get_message(self):
        return str.join("\n", [f"**{self.title}**"] + self.lines)

    async def display_report(self, title: str, report: RunResultReport):
        raise NotImplementedError()

    async def _update_message(self):
        raise NotImplementedError()


class RunProgressReporterDiscord(RunProgressReporter):
    def __init__(
        self,
        root: MultiProgressReporter,
        interaction: discord.Interaction,
        title: str,
    ):
        super().__init__(title=title)
        self.root = root
        self.interaction = interaction

    async def _update_message(self):
        await self.root._update_message()

    async def display_report(self, title: str, report: RunResultReport):
        thread = await self.interaction.channel.create_thread(
            name=title,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
        )
        await thread.add_user(self.interaction.user)
        message = ""
        for part in report.data:
            if isinstance(part, Text):
                if len(message) + len(part.text) > 1900:
                    await thread.send(message)
                    message = ""
                message += part.text
            elif isinstance(part, Log):
                message = await _send_split_log(thread, message, part.header, part.content)

        if len(message) > 0:
            await thread.send(message)

        await self.push(f"See results at {thread.jump_url}")


class RunProgressReporterAPI(RunProgressReporter):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.long_report = ""

    async def _update_message(self):
        pass

    async def display_report(self, title: str, report: RunResultReport):
        for part in report.data:
            if isinstance(part, Text):
                self.long_report += part.text
            elif isinstance(part, Log):
                self.long_report += f"\n\n## {part.header}:\n"
                self.long_report += f"```\n{part.content}```"
