"""Custom reward names work in both CLI dispatch and weighted mixtures."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import importlib
import sys
from argparse import Namespace
from unittest.mock import AsyncMock, Mock

import pytest

import miles.rollout.rm_hub.api as api_module
import miles.rollout.rm_hub.registry as registry_module
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub.api import ApiReward
from miles.rollout.rm_hub.registry import get_reward_registry
from miles.utils.api_rm_config import ApiRewardConfig
from miles.utils.types import Sample


@pytest.fixture
def custom_registry(tmp_path, monkeypatch):
    module_name = "custom_reward_registry"
    (tmp_path / f"{module_name}.py").write_text(
        "async def prompt_reward(args, samples):\n"
        "    return [float(sample.prompt) for sample in samples]\n"
        "REWARDS = {'prompt_score': prompt_reward}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(module_name)
    args = Namespace(custom_rm_registry_path=f"{module_name}.REWARDS", custom_rm_path=None, rm_type="prompt_score")
    yield args, module
    sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_imported_reward_is_available_to_single_and_batched_dispatch(custom_registry):
    args, module = custom_registry
    samples = [Sample(prompt="-7"), Sample(prompt="12.5")]

    registry = get_reward_registry(args)
    assert set(get_reward_registry()) <= registry.keys()
    assert registry["prompt_score"] is module.prompt_reward
    assert "prompt_score" not in get_reward_registry()
    assert await async_rm(args, samples[0]) == -7.0
    assert await batched_async_rm(args, samples) == [-7.0, 12.5]


@pytest.mark.asyncio
async def test_sample_metadata_selects_registered_rewards_in_order(custom_registry):
    args, module = custom_registry
    alternate = AsyncMock(return_value=[4.0])
    module.REWARDS["alternate"] = alternate
    samples = [
        Sample(prompt="2"),
        Sample(prompt="3", metadata={"rm_type": "alternate"}),
        Sample(prompt="5"),
    ]

    assert await batched_async_rm(args, samples) == [2.0, 4.0, 5.0]
    alternate.assert_awaited_once_with(args, [samples[1]])


@pytest.mark.asyncio
async def test_custom_rm_path_keeps_precedence(custom_registry):
    args, _ = custom_registry
    args.custom_rm_path = "custom_reward_registry.prompt_reward"
    args.custom_rm_registry_path = "does.not.exist"
    args.rm_type = "unknown"
    assert await batched_async_rm(args, [Sample(prompt="3")]) == [3.0]


@pytest.mark.parametrize(
    "custom_rewards, error, message",
    [
        ([], TypeError, "mapping"),
        ({"hps": AsyncMock()}, ValueError, "reserved"),
        ({"weighted": AsyncMock()}, ValueError, "reserved"),
        ({"bad,name": AsyncMock()}, ValueError, "Invalid custom reward name"),
        ({"judge": 42}, TypeError, "reward callable"),
    ],
)
def test_invalid_registry_is_rejected(custom_registry, custom_rewards, error, message):
    args, module = custom_registry
    module.REWARDS = custom_rewards
    with pytest.raises(error, match=message):
        get_reward_registry(args)


@pytest.mark.asyncio
async def test_two_api_configs_mix_with_hps_and_reuse_separate_pools(custom_registry, monkeypatch):
    args, module = custom_registry
    actor_class = "tests.fast.rollout.test_api_reward_pool.CustomApiRewardActor"
    configs = [
        ApiRewardConfig(
            actor_class=actor_class, actor_kwargs={"endpoint": endpoint, "timeout_s": 5}, max_concurrency=2
        )
        for endpoint in ["http://first.test", "http://second.test"]
    ]
    first = ApiReward("first", configs[0])
    second = ApiReward("second", configs[1])
    module.REWARDS = {"first": first, "second": second}
    pools = [AsyncMock(), AsyncMock()]
    pools[0].score.return_value = ([1.0, 2.0], 1)
    pools[1].score.return_value = ([4.0, 8.0], 3)
    create_pool = Mock(side_effect=pools)
    monkeypatch.setattr(api_module, "AsyncRewardActorPool", create_pool)
    hps_rm = AsyncMock(return_value=[0.1, 0.2])
    monkeypatch.setitem(registry_module._BUILTIN_REWARDS, "hps", hps_rm)
    args.custom_rm_path = "miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm"
    args.custom_rm_args = "hps=0.5,first=0.2,second=0.3"
    args.reward_key = "weighted"
    samples = [Sample(prompt="one"), Sample(prompt="two")]

    # Importing the registry and scoring an empty batch should not start Ray workers.
    get_reward_registry(args)
    assert await first(args, []) == []
    create_pool.assert_not_called()

    rewards = await batched_async_rm(args, samples)
    assert [(r["hps"], r["first"], r["second"]) for r in rewards] == [(0.1, 1.0, 4.0), (0.2, 2.0, 8.0)]
    assert [r["weighted"] for r in rewards] == pytest.approx([1.45, 2.9])
    assert all(s.reward_max_queue_depth == {"first": 1.0, "second": 3.0} for s in samples)
    hps_rm.assert_awaited_once_with(args, samples)
    assert create_pool.call_count == 2
    pool_kwargs = [call.kwargs for call in create_pool.call_args_list]
    assert pool_kwargs[0]["actor_cls"] is pool_kwargs[1]["actor_cls"]
    assert [kw["actor_kwargs"] for kw in pool_kwargs] == [config.actor_kwargs for config in configs]
    assert [kw["name"] for kw in pool_kwargs] == ["first", "second"]
    assert all(kw["num_gpus_per_worker"] == 0 and kw["actor_max_concurrency"] == 2 for kw in pool_kwargs)

    # The standalone CLI entry finds the same configured instance used by the mixture.
    args.custom_rm_path = None
    for name, scores in [("first", [1.0, 2.0]), ("second", [4.0, 8.0])]:
        args.rm_type = name
        assert await batched_async_rm(args, samples) == scores
    assert create_pool.call_count == 2
    assert all(pool.score.await_count == 2 for pool in pools)
