"""API scorer serialization, validation, and the shared reward dispatch contract."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import base64
import io
import json
import pickle
from argparse import Namespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
import pytest_asyncio
import torch
import yaml
from PIL import Image

import miles.rollout.rm_hub.api as api_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.api import (
    ApiRewardActor,
    ApiRewardConfig,
    api_rm,
    close_api_rm_pools,
    get_api_rm_configs,
    load_api_rm_configs,
)
from miles.utils.types import Sample


def _config(**overrides):
    return ApiRewardConfig(model="judge-v1", api_key_env="TEST_RM_KEY", **overrides)


def _args(**configs):
    return Namespace(rm_type=next(iter(configs)), custom_rm_path=None, _api_rm_configs=configs)


def _sample(index):
    return Sample(index=index, prompt=str(index), generated_output=torch.full((3, 1, 8, 8), index / 4))


def _response(score=1, *, content=None):
    return httpx.Response(
        200,
        json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "judge-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"score": score}) if content is None else content,
                        "refusal": None,
                    },
                }
            ],
        },
    )


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(monkeypatch):
    monkeypatch.setenv("TEST_RM_KEY", "test-only-secret")
    yield
    await close_api_rm_pools()


@pytest.fixture
def sdk_transport(monkeypatch):
    original = openai.OpenAI
    created, clients = [], []

    def install(handler):
        def factory(**kwargs):
            created.append(kwargs)
            client = original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client

        monkeypatch.setattr(openai, "OpenAI", factory)
        return created

    yield install
    for client in clients:
        client.close()


def test_actor_preserves_image_prompt_pairing_and_closes_client(sdk_transport):
    def handler(request):
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        index = int(content[0]["text"])
        data_url = content[1]["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        assert image.getpixel((0, 0)) == (round(index / 4 * 255),) * 3
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["model"] == "judge-v1"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        return _response(index)

    clients = sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    samples = [_sample(2), _sample(1)]
    assert actor.score_batch([s.generated_output for s in samples], [s.prompt for s in samples]) == [2.0, 1.0]
    assert clients[0]["max_retries"] == 0
    actor.close()
    assert actor.scorer.client.is_closed()


async def test_rm_reuses_pools_by_alias_and_records_queue_depth(monkeypatch):
    created = {}

    def make_pool(name, config):
        pool = AsyncMock()
        pool.score.return_value = ([1.0], 3 if name == "judge" else 2)
        assert name not in created
        created[name] = pool
        return pool

    monkeypatch.setattr(api_module, "AsyncApiRewardPool", make_pool)
    args = _args(judge=_config(), other=_config())
    sample = _sample(1)
    for name in ("judge", "other", "judge"):
        assert await api_rm(args, [sample], name=name) == [1.0]
    assert created["judge"].score.await_count == 2
    (output,), prompts = created["judge"].score.await_args.args
    assert output is sample.generated_output
    assert prompts == [sample.prompt]
    assert sample.reward_max_queue_depth == {"judge": 3.0, "other": 2.0}
    await close_api_rm_pools()
    for pool in created.values():
        pool.close.assert_awaited_once()
    assert not api_module._pools


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_http_failure_is_fatal_without_sdk_retries(status, sdk_transport):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "test failure"}})

    sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    with pytest.raises(openai.APIStatusError):
        actor.score_batch([_sample(1).generated_output], ["1"])
    with pytest.raises(RuntimeError, match="stopped after a scoring failure"):
        actor.score_batch([_sample(2).generated_output], ["2"])
    assert calls == 1


@pytest.mark.parametrize(
    "response_kwargs",
    [
        {"content": ""},
        {"content": "Score: 2"},
        {"content": '{"score": "2"}'},
        {"content": '{"score": true}'},
        {"content": '{"score": NaN}'},
        {"content": '{"score": Infinity}'},
        {"content": "{}"},
        {"content": "[]"},
        {"score": -1},
        {"score": 5},
    ],
)
def test_invalid_response_never_becomes_a_reward(response_kwargs, sdk_transport):
    sdk_transport(lambda request: _response(**response_kwargs))
    actor = ApiRewardActor(config=_config())
    with pytest.raises((ValueError, KeyError, TypeError)):
        actor.score_batch([_sample(1).generated_output], ["1"])


@pytest.mark.parametrize("output", [None, torch.zeros(3, 2, 8, 8), torch.zeros(1, 16000)])
def test_unsupported_media_fails_before_http(output, sdk_transport):
    def handler(request):
        pytest.fail("Unsupported media must not be sent to the API")

    sdk_transport(handler)
    actor = ApiRewardActor(config=_config())
    with pytest.raises((AttributeError, ValueError)):
        actor.score_batch([output], ["1"])


async def test_builtin_dispatch_and_per_sample_override(monkeypatch):
    pool = AsyncMock()
    pool.score.side_effect = [([1.0, 2.0], 0), ([3.0], 0)]
    monkeypatch.setattr(api_module, "AsyncApiRewardPool", lambda name, config: pool)
    args = _args(judge=_config())
    assert await batched_async_rm(args, [_sample(1), _sample(2)]) == [1.0, 2.0]
    args.rm_type = "unused"
    sample = _sample(3)
    sample.metadata = {"rm_type": "judge"}
    assert await batched_async_rm(args, [sample]) == [3.0]


def test_config_prompt_resolution_and_no_credentials_in_serialized_args(tmp_path):
    (tmp_path / "rubric.txt").write_text("Evaluate prompt adherence from 0 to 10. Return JSON with score.")
    config_path = tmp_path / "rm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "judge": {
                    "model": "judge-version-123",
                    "api_key_env": "TEST_RM_KEY",
                    "prompt_path": "rubric.txt",
                    "score_max": 10,
                }
            }
        )
    )
    args = Namespace(api_rm_config=str(config_path))
    get_api_rm_configs(args)
    assert get_api_rm_configs(args)["judge"].model == "judge-version-123"
    (tmp_path / "rubric.txt").unlink()
    assert "0 to 10" in get_api_rm_configs(args)["judge"].prompt
    assert b"test-only-secret" not in pickle.dumps(args)


@pytest.mark.parametrize("entry", [{"max_concurrency": 0}, {"api_key": "not-allowed"}])
def test_invalid_config_is_rejected(tmp_path, entry):
    path = tmp_path / "rm.yaml"
    path.write_text(yaml.safe_dump({"judge": {"model": "judge", "api_key_env": "TEST_RM_KEY", **entry}}))
    with pytest.raises(ValueError):
        load_api_rm_configs(str(path))


def test_api_alias_cannot_shadow_local_reward(tmp_path):
    path = tmp_path / "rm.yaml"
    path.write_text("hps:\n  model: judge\n  api_key_env: TEST_RM_KEY\n")
    with pytest.raises(ValueError, match="reserved API reward name"):
        load_api_rm_configs(str(path))
