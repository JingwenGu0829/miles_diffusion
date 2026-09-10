"""API pool worker configuration, without starting Ray or an HTTP server.

Mental model: max_concurrency=2 -> two zero-GPU actors, one request per actor call.
The shared pool's placement rules are covered by test_reward_pool_placement.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from unittest.mock import Mock, call

import miles.rollout.rm_hub.core as core_module
from miles.rollout.rm_hub.api import ApiRewardActor, AsyncApiRewardPool
from miles.utils.api_rm_config import ApiRewardConfig


def test_pool_uses_configured_zero_gpu_workers(monkeypatch):
    actor_cls = Mock()
    actor_cls.options.return_value = actor_cls
    actor_cls.remote.side_effect = lambda **kwargs: Mock()
    remote = Mock(return_value=actor_cls)
    monkeypatch.setattr(core_module.ray, "remote", remote)
    config = ApiRewardConfig(model="judge", api_key_env="TEST_RM_KEY", max_concurrency=2)
    pool = AsyncApiRewardPool("judge", config)

    assert remote.call_args_list == [call(ApiRewardActor)] * 2
    assert (
        remote.return_value.options.call_args_list == [call(num_cpus=0, num_gpus=0, scheduling_strategy="DEFAULT")] * 2
    )
    assert remote.return_value.remote.call_args_list == [call(config=config)] * 2
    assert pool._batch_size == 1
