"""API pool wiring and lifecycle, without starting Ray or an HTTP server.

Mental model: max_concurrency=2 -> two zero-GPU actors, one request per actor call.
Covered: worker options and normal client cleanup; a failed/cancelled score closes its pool.
The shared pool's placement rules are covered by test_reward_pool_placement.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
from unittest.mock import AsyncMock, Mock, call

import pytest

import miles.rollout.rm_hub.api as api_module
from miles.rollout.rm_hub.api import ApiRewardActor, ApiRewardConfig, AsyncApiRewardPool
from miles.rollout.rm_hub.core import AsyncRewardActorPool


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    actor_cls = Mock()
    actor_cls.options.return_value = actor_cls
    actor_cls.remote.side_effect = lambda **kwargs: Mock()
    monkeypatch.setattr(api_module.ray, "remote", Mock(return_value=actor_cls))
    monkeypatch.setattr(api_module.ray, "get", Mock())
    monkeypatch.setattr(api_module.ray, "kill", Mock())


@pytest.mark.asyncio
async def test_pool_uses_zero_gpu_workers_and_closes_each_client():
    config = ApiRewardConfig(model="judge", api_key_env="TEST_RM_KEY", max_concurrency=2)
    pool = AsyncApiRewardPool("judge", config)
    remote = api_module.ray.remote

    assert remote.call_args_list == [call(ApiRewardActor)] * 2
    assert (
        remote.return_value.options.call_args_list == [call(num_cpus=0, num_gpus=0, scheduling_strategy="DEFAULT")] * 2
    )
    assert remote.return_value.remote.call_args_list == [call(config=config)] * 2
    assert pool._batch_size == 1

    await pool.close()

    for actor in pool._actors:
        actor.close.remote.assert_called_once_with()
    api_module.ray.get.assert_called_once_with([actor.close.remote.return_value for actor in pool._actors])
    assert api_module.ray.kill.call_args_list == [call(actor, no_restart=True) for actor in pool._actors]
    await pool.close()
    assert api_module.ray.kill.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("judge unavailable"), asyncio.CancelledError()])
async def test_failed_or_cancelled_score_closes_pool(monkeypatch, error):
    score = AsyncMock(side_effect=error)
    monkeypatch.setattr(AsyncRewardActorPool, "score", score)
    pool = AsyncApiRewardPool("judge", ApiRewardConfig(model="judge", api_key_env="TEST_RM_KEY", max_concurrency=2))

    with pytest.raises(type(error)):
        await pool.score(["output"], ["prompt"])

    assert api_module.ray.kill.call_args_list == [call(actor, no_restart=True) for actor in pool._actors]
    with pytest.raises(RuntimeError, match="pool is closed"):
        await pool.score(["output"], ["prompt"])
    score.assert_awaited_once_with(["output"], ["prompt"])
