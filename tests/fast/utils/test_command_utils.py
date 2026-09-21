"""`_cvd_export` turns the recipe's device list into a `ray start` prefix:

    unset        -> ""                                     inherit the environment
    "4,5,2", 3   -> "export CUDA_VISIBLE_DEVICES=4,5,2 && " pin the raylet
    "0,1",   5   -> AssertionError                          ray would hand out unknown ids

Runtime env values travel through a private file; logs redact only explicitly named variables.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils import command_utils as commands
from miles.utils.external_utils.command_utils import ExecuteTrainConfig, _cvd_export


def test_unset_inherits_the_environment():
    assert _cvd_export(ExecuteTrainConfig(), num_gpus_per_node=4) == ""


def test_set_exports_for_ray_start():
    config = ExecuteTrainConfig(cuda_visible_devices="4,5,2")
    assert _cvd_export(config, num_gpus_per_node=3) == "export CUDA_VISIBLE_DEVICES=4,5,2 && "


def test_count_mismatch_is_rejected():
    config = ExecuteTrainConfig(cuda_visible_devices="0,1")
    with pytest.raises(AssertionError, match="lists 2 GPU"):
        _cvd_export(config, num_gpus_per_node=5)


@pytest.mark.parametrize("redact_env_vars", [(), ("TEST_RM_KEY",)])
def test_submit_logs_only_selected_env_values_redacted(monkeypatch, capsys, redact_env_vars):
    monkeypatch.delenv("TEST_RM_KEY", raising=False)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setattr(commands, "check_has_nvlink", lambda: False)
    submissions = []
    submitted_envs = []

    def execute(argv, **kwargs):
        command = argv[2]
        assert "test-secret-not-for-logs" not in command
        if "ray job submit" not in command:
            return subprocess.CompletedProcess(argv, 0)
        tokens = shlex.split(command)
        runtime_path = Path(next(t.split("=", 1)[1] for t in tokens if t.startswith("--runtime-env=")))
        assert runtime_path.stat().st_mode & 0o777 == 0o600
        env = json.loads(runtime_path.read_text())["env_vars"]
        assert env["TEST_RM_KEY"] == "test-secret-not-for-logs"
        submissions.append(runtime_path)
        submitted_envs.append(env)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    commands.execute_train(
        "--api-rm-config unused.yaml --rm-type api",
        1,
        config=ExecuteTrainConfig(extra_env_vars='{"TEST_RM_KEY": "test-secret-not-for-logs"}'),
        extra_env_vars={"TEST_RM_KEY": "overridden-secret", "OTHER_API_KEY": "unselected-test-value"},
        redact_env_vars=redact_env_vars,
    )
    assert len(submissions) == 1
    assert not submissions[0].exists()
    output = capsys.readouterr().out
    log_line = next(line for line in output.splitlines() if line.startswith("Runtime env: "))
    logged_env = json.loads(log_line.removeprefix("Runtime env: "))["env_vars"]
    expected_env = dict(submitted_envs[0])
    if redact_env_vars:
        expected_env["TEST_RM_KEY"] = "***"
        assert "test-secret-not-for-logs" not in output
    assert logged_env == expected_env
    assert logged_env["NCCL_DEBUG"] == "INFO"
    assert logged_env["OTHER_API_KEY"] == "unselected-test-value"
    assert "overridden-secret" not in output
