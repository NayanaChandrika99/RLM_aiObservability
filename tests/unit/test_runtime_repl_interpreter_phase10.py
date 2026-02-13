# ABOUTME: Validates the Phase 10 subprocess sandbox behavior for REPL code execution.
# ABOUTME: Ensures blocked imports, timeout enforcement, and output truncation are deterministic.

from __future__ import annotations

from investigator.runtime.repl_interpreter import execute_in_sandbox


def test_execute_in_sandbox_blocks_dangerous_module_import() -> None:
    result = execute_in_sandbox("import os\nprint(os.getcwd())")

    assert result.returncode != 0
    combined = f"{result.stdout}\n{result.stderr}".lower()
    assert "importerror" in combined or "blocked" in combined


def test_execute_in_sandbox_enforces_timeout() -> None:
    result = execute_in_sandbox("while True:\n    pass", timeout_sec=1)

    assert result.returncode != 0
    assert result.timed_out is True
    assert "timeout" in result.stderr.lower()


def test_execute_in_sandbox_truncates_stdout() -> None:
    stdout_payload = "x" * 12000
    code = f"print({stdout_payload!r})"
    result = execute_in_sandbox(code, max_output_chars=8192)

    assert result.returncode == 0
    assert len(result.stdout) <= 8192


def test_execute_in_sandbox_truncates_stderr() -> None:
    stderr_payload = "y" * 12000
    result = execute_in_sandbox(f"raise RuntimeError({stderr_payload!r})", max_output_chars=8192)

    assert result.returncode != 0
    assert len(result.stderr) <= 8192
