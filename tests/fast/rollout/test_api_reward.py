"""API reward contract: real SDK serialization, ordered scores, and fatal failures."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import base64
import io
import json
import pickle
from argparse import Namespace

import httpx
import openai
import pytest
import pytest_asyncio
import torch
import yaml
from PIL import Image

from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.api import (
    ApiRewardConfig,
    api_rm,
    close_api_rm_clients,
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


def _response(score=1, *, content=None, **kwargs):
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
        **kwargs,
    )


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(monkeypatch):
    monkeypatch.setenv("TEST_RM_KEY", "test-only-secret")
    yield
    await close_api_rm_clients()


@pytest.fixture
def sdk_transport(monkeypatch):
    original = openai.AsyncOpenAI
    created = []

    def install(handler):
        def factory(**kwargs):
            created.append(kwargs)
            return original(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(openai, "AsyncOpenAI", factory)
        return created

    return install


async def test_order_image_identity_and_shared_concurrency_across_microgroups(sdk_transport):
    active = peak = 0
    completed = []

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
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
        await asyncio.sleep(0.05 if index == 0 else 0.005)
        completed.append(index)
        active -= 1
        return _response(index)

    clients = sdk_transport(handler)
    args = _args(judge=_config(max_concurrency=2))
    first, second = await asyncio.gather(
        api_rm(args, [_sample(0), _sample(1)]),
        api_rm(args, [_sample(2), _sample(3)]),
    )
    assert first == [0.0, 1.0]
    assert second == [2.0, 3.0]
    assert completed[0] != 0
    assert peak == 2
    assert len(clients) == 1
    assert clients[0]["max_retries"] == 0


async def test_all_local_rewards_mix_with_two_api_configs_without_crossing_samples(monkeypatch, sdk_transport):
    import miles.rollout.rm_hub.weighted_mixture_rm as mixture

    async def local(args, samples):
        return [sample.index / 10 for sample in samples]

    monkeypatch.setattr(mixture, "_REWARDS", {name: local for name in ("hps", "pickscore", "ocr")})
    seen = set()

    async def handler(request):
        payload = json.loads(request.content)
        model = payload["model"]
        index = int(payload["messages"][1]["content"][0]["text"])
        seen.add((str(request.url), model))
        await asyncio.sleep(0.01 if index == 1 else 0)
        return _response(index if model == "gpt-test" else 4 - index)

    sdk_transport(handler)
    args = _args(
        openai=_config().model_copy(update={"model": "gpt-test"}),
        gemini=_config(base_url="https://generativelanguage.googleapis.com/v1beta/openai/").model_copy(
            update={"model": "gemini-test"}
        ),
    )
    args.custom_rm_args = "hps=0.2,pickscore=0.3,ocr=0.4,openai=0.5,gemini=0.6"
    args.reward_key = "weighted"
    rewards = await mixture.weighted_mixture_rm(args, [_sample(1), _sample(2)])
    for index, reward in zip((1, 2), rewards, strict=True):
        assert reward["openai"] == index
        assert reward["gemini"] == 4 - index
        assert reward["weighted"] == pytest.approx(0.9 * index / 10 + 0.5 * index + 0.6 * (4 - index))
    assert seen == {
        ("https://api.openai.com/v1/chat/completions", "gpt-test"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-test"),
    }


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_http_failure_is_fatal_without_sdk_retries(status, sdk_transport):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "test failure"}})

    sdk_transport(handler)
    with pytest.raises(RuntimeError, match="sample index=1"):
        await api_rm(_args(judge=_config()), [_sample(1)])
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
async def test_invalid_or_incomplete_response_never_becomes_a_reward(response_kwargs, sdk_transport):
    sdk_transport(lambda request: _response(**response_kwargs))
    with pytest.raises(RuntimeError):
        await api_rm(_args(judge=_config()), [_sample(1)])


async def test_failure_cancels_other_requests(sdk_transport):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        index = json.loads(request.content)["messages"][1]["content"][0]["text"]
        if index == "1":
            await started.wait()
            return httpx.Response(500, json={"error": {"message": "failed"}})
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    sdk_transport(handler)
    with pytest.raises(RuntimeError):
        await api_rm(_args(judge=_config()), [_sample(1), _sample(2)])
    assert cancelled.is_set()


@pytest.mark.parametrize("output", [None, torch.zeros(3, 2, 8, 8), torch.zeros(1, 16000)])
async def test_unsupported_media_fails_before_http(output, sdk_transport):
    def handler(request):
        pytest.fail("Unsupported media must not be sent to the API")

    sdk_transport(handler)
    sample = _sample(1)
    sample.generated_output = output
    with pytest.raises(RuntimeError):
        await api_rm(_args(judge=_config()), [sample])


async def test_builtin_dispatch_and_per_sample_override(sdk_transport):
    sdk_transport(lambda request: _response(int(json.loads(request.content)["messages"][1]["content"][0]["text"])))
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


@pytest.mark.parametrize(
    "entry",
    [
        {"max_concurrency": 0},
        {"api_key": "not-allowed"},
    ],
)
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
