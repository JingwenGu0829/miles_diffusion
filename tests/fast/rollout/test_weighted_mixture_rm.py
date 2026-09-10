"""weighted_mixture_rm combines built-in rewards with weights parsed from --custom-rm-args.

Mental model (--custom-rm-args "hps=0.7,pickscore=0.3" --reward-key weighted, one batch of 2 samples):

    weighted_mixture_rm ─┬─ hps_rm(args, [s0, s1])       -> [0.3, 0.2]  x 0.7
                         └─ pickscore_rm(args, [s0, s1]) -> [0.8, 0.9]  x 0.3
                         = [{hps: 0.3, pickscore: 0.8, weighted: 0.45}, {hps: 0.2, pickscore: 0.9, weighted: 0.41}]

Covered: each reward scores the whole batch once and every sample gets its components plus the
weighted sum (1); an unknown reward name in --custom-rm-args is rejected (2); a --reward-key
that names neither a component nor "weighted" is rejected before any reward runs (3);
score counts must match the batch (4); local and API rewards can be mixed without alias crosstalk (5).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from unittest.mock import AsyncMock

import pytest

import miles.rollout.rm_hub.api as api_module
import miles.rollout.rm_hub.weighted_mixture_rm as weighted_mixture_rm_module
from miles.rollout.rm_hub.api import ApiRewardConfig
from miles.rollout.rm_hub.weighted_mixture_rm import parse_weights, weighted_mixture_rm
from miles.utils.types import Sample


def _fake_rewards(calls):
    def fake(scores):
        async def rm(args, samples):
            calls.append(len(samples))
            return scores

        return rm

    return {"hps": fake([0.3, 0.2]), "pickscore": fake([0.8, 0.9])}


@pytest.mark.asyncio
async def test_each_sample_gets_its_components_and_the_weighted_sum(monkeypatch):
    """Fanning the batch out per sample, dropping a weight, or collapsing to a scalar would all show here."""
    calls = []
    monkeypatch.setattr(weighted_mixture_rm_module, "_REWARDS", _fake_rewards(calls))
    args = Namespace(_api_rm_configs={}, custom_rm_args="hps=0.7,pickscore=0.3", reward_key="weighted")

    rewards = await weighted_mixture_rm(args, [object(), object()])

    assert [r["weighted"] for r in rewards] == pytest.approx([0.45, 0.41])
    assert [(r["hps"], r["pickscore"]) for r in rewards] == [(0.3, 0.8), (0.2, 0.9)]
    assert calls == [2, 2]


def test_unknown_reward_name_is_rejected():
    with pytest.raises(ValueError, match="unknown reward 'clip'"):
        parse_weights("hps=0.7,clip=0.3")


@pytest.mark.asyncio
async def test_missing_reward_key_is_rejected_before_scoring(monkeypatch):
    calls = []
    monkeypatch.setattr(weighted_mixture_rm_module, "_REWARDS", _fake_rewards(calls))

    with pytest.raises(ValueError, match="--reward-key weighted"):
        await weighted_mixture_rm(
            Namespace(_api_rm_configs={}, custom_rm_args="hps=0.7,pickscore=0.3", reward_key=None), [object()]
        )
    assert calls == []


@pytest.mark.asyncio
async def test_wrong_score_count_is_rejected_instead_of_dropping_samples(monkeypatch):
    async def wrong_length(args, samples):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(weighted_mixture_rm_module, "_REWARDS", {"hps": wrong_length})
    args = Namespace(_api_rm_configs={}, custom_rm_args="hps=1", reward_key="weighted")
    with pytest.raises(ValueError, match="returned 3 scores for 2 samples"):
        await weighted_mixture_rm_module.weighted_mixture_rm(args, [object(), object()])


@pytest.mark.asyncio
async def test_local_and_api_rewards_mix_without_alias_crosstalk(monkeypatch):
    """Keep real name resolution; only the expensive scorers/pools are replaced."""
    hps_rm = AsyncMock(return_value=[0.1, 0.2])
    monkeypatch.setitem(weighted_mixture_rm_module._REWARDS, "hps", hps_rm)
    pools = {name: AsyncMock() for name in ("judge", "reverse")}
    pools["judge"].score.return_value = ([1.0, 2.0], 0)
    pools["reverse"].score.return_value = ([3.0, 2.0], 0)
    monkeypatch.setattr(api_module, "_pools", pools)
    args = Namespace(
        _api_rm_configs={name: ApiRewardConfig(model=name, api_key_env="TEST_RM_KEY") for name in pools},
        custom_rm_args="hps=0.2,judge=0.5,reverse=0.3",
        reward_key="weighted",
    )
    samples = [Sample(prompt="first"), Sample(prompt="second")]

    rewards = await weighted_mixture_rm(args, samples)

    assert [r["weighted"] for r in rewards] == pytest.approx([1.42, 1.64])
    assert [(r["judge"], r["reverse"]) for r in rewards] == [(1.0, 3.0), (2.0, 2.0)]
    hps_rm.assert_awaited_once_with(args, samples)
    for pool in pools.values():
        pool.score.assert_awaited_once_with([None, None], ["first", "second"])
