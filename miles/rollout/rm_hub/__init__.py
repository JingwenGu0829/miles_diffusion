import asyncio

from miles.utils.misc import load_function
from miles.utils.types import Sample

from .registry import get_reward_registry


def _resolve_rm_type(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (metadata.get("rm_type") or args.rm_type or "").strip()


async def async_rm(args, sample: Sample, **kwargs):
    rm_type = _resolve_rm_type(args, sample)

    reward = get_reward_registry(args).get(rm_type)
    if reward is None:
        raise NotImplementedError(f"Rule-based RM for {rm_type!r} is not implemented.")
    return (await reward(args, [sample]))[0]


def create_colocated_reward_pools(args, placement_group, slots) -> list:
    """Seat every colocated pool; the rm functions' singleton lookup then finds them."""
    pools = []
    if args.pickscore_reward_colocate:
        from .pickscore import AsyncPickScorePool

        pools.append(AsyncPickScorePool(args, placement_group=placement_group, slots=slots))
    if args.hps_reward_colocate:
        from .hps import AsyncHPSPool

        pools.append(AsyncHPSPool(args, placement_group=placement_group, slots=slots))
    return pools


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)

    if samples:
        rm_types = [_resolve_rm_type(args, sample) for sample in samples]
        if all(rm_type == rm_types[0] for rm_type in rm_types):
            reward = get_reward_registry(args).get(rm_types[0])
            if reward is not None:
                return await reward(args, samples)

    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
