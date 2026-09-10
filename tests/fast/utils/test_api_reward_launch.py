"""API credentials reach Ray workers without appearing in logged shell commands."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import json
import shlex
from pathlib import Path

import pytest

from miles.utils.external_utils import command_utils as commands


def _config(tmp_path):
    path = tmp_path / "reward config.yaml"
    path.write_text("model: gemini-3.8-flash\napi_key_env: TEST_GEMINI_KEY\n")
    return path


def test_submit_passes_configured_key_in_private_runtime_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_GEMINI_KEY", "test-secret-not-for-logs")
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setattr(commands, "check_has_nvlink", lambda: False)
    submissions = []

    def execute(command, **kwargs):
        assert "test-secret-not-for-logs" not in command
        if "ray job submit" not in command:
            return ""
        tokens = shlex.split(command)
        runtime_path = Path(next(t.split("=", 1)[1] for t in tokens if t.startswith("--runtime-env=")))
        assert runtime_path.stat().st_mode & 0o777 == 0o600
        env = json.loads(runtime_path.read_text())["env_vars"]
        assert env["TEST_GEMINI_KEY"] == "test-secret-not-for-logs"
        submissions.append(runtime_path)
        return ""

    monkeypatch.setattr(commands, "exec_command", execute)
    commands.execute_train(f"--api-rm-config {shlex.quote(str(_config(tmp_path)))} --rm-type api", 1)
    assert len(submissions) == 1
    assert not submissions[0].exists()


def test_missing_key_fails_before_any_cluster_commands(monkeypatch, tmp_path):
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)
    monkeypatch.setattr(commands, "exec_command", lambda *a, **kw: pytest.fail("Must validate before cluster changes"))
    with pytest.raises(KeyError, match="TEST_GEMINI_KEY"):
        commands.execute_train(f"--api-rm-config={shlex.quote(str(_config(tmp_path)))}", 1)


def test_launch_without_api_config_does_not_require_keys():
    assert commands._api_rm_env_vars("--rm-type hps") == {}
