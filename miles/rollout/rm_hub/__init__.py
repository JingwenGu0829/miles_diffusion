import asyncio
from functools import partial

from miles.utils.misc import load_function
from miles.utils.types import Sample

BUILTIN_REWARDS = {
    "ocr": "miles.rollout.rm_hub.ocr.ocr_rm",
    "pickscore": "miles.rollout.rm_hub.pickscore.pickscore_rm",
    "hps": "miles.rollout.rm_hub.hps.hps_rm",
}


def resolve_reward(args, name: str):
    """Resolve every reward to the same async callable(args, samples) contract."""
    if name in BUILTIN_REWARDS:
        return load_function(BUILTIN_REWARDS[name])

    from .api import api_rm, get_api_rm_configs

    if name in get_api_rm_configs(args):
        return partial(api_rm, name=name)
    raise NotImplementedError(f"Rule-based RM for {name!r} is not implemented.")


def _resolve_rm_type(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (metadata.get("rm_type") or args.rm_type or "").strip()


async def async_rm(args, sample: Sample, **kwargs):
    rm_function = resolve_reward(args, _resolve_rm_type(args, sample))
    return (await rm_function(args, [sample]))[0]


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
        if len(set(rm_types)) == 1:
            rm_function = resolve_reward(args, rm_types[0])
            return await rm_function(args, samples)

    return await asyncio.gather(*(async_rm(args, sample, **kwargs) for sample in samples))
