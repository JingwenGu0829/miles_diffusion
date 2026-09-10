"""``--custom-rm-path`` example: a weighted sum of local and API rewards, weighted by ``--custom-rm-args``.

    --custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \\
    --custom-rm-args "hps=0.7,pickscore=0.3" --reward-key weighted

To include an API reward, configure the ``judge`` alias in ``rewards.yaml``
(see ``docs/user-guide/rewards.md``), then use:

    --api-rm-config rewards.yaml \\
    --custom-rm-path miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm \\
    --custom-rm-args "hps=0.7,judge=0.3" --reward-key weighted

Each sample's reward is a dict holding every component plus ``"weighted"``, so each reward
gets its own ``rollout/reward/<name>_mean`` panel while ``--reward-key`` picks what GRPO trains
on. ``weighted`` selects the returned dictionary entry; this function computes the sum.
Each named reward scores the whole batch once. Local rewards keep their own placement flags
(``--<rm>-reward-colocate``, ``--<rm>-num-gpus-per-worker``). Weights apply to raw scores,
whose scales differ: HPSv2.1 ~0.3, PickScore/26 ~0.85, OCR in [0, 1], default API rubric in [0, 4].
API rewards use their YAML settings and do not consume local GPU reward slots.
"""

import asyncio
from collections.abc import Sequence

from miles.utils.types import Sample

from . import BUILTIN_REWARDS, resolve_reward
from .api import get_api_rm_configs


def parse_weights(custom_rm_args: str, api_names: Sequence[str] = ()) -> list[tuple[str, float]]:
    weights = []
    # launch scripts hand the arg string to `sh`, where ";" would end the command; "," is inert
    for term in custom_rm_args.split(","):
        name, _, weight = term.strip().partition("=")
        if name not in BUILTIN_REWARDS and name not in api_names:
            raise ValueError(
                f"--custom-rm-args: unknown reward {name!r} in {custom_rm_args!r}; "
                f"choose from {(*BUILTIN_REWARDS, *api_names)}"
            )
        weights.append((name, float(weight)))
    return weights


async def weighted_mixture_rm(args, samples: Sequence[Sample], **kwargs) -> list[dict[str, float]]:
    weights = parse_weights(args.custom_rm_args, tuple(get_api_rm_configs(args)))
    if args.reward_key not in {name for name, _ in weights} | {"weighted"}:
        raise ValueError(
            f"weighted_mixture_rm returns a dict per sample; pass --reward-key weighted (or one of "
            f"{[name for name, _ in weights]}), got {args.reward_key!r}"
        )
    rm_functions = [resolve_reward(args, name) for name, _ in weights]
    per_reward = await asyncio.gather(*(rm_function(args, samples) for rm_function in rm_functions))
    for (name, _), scores in zip(weights, per_reward, strict=True):
        if len(scores) != len(samples):
            raise ValueError(f"Reward {name!r} returned {len(scores)} scores for {len(samples)} samples")
    rewards = []
    for i in range(len(samples)):
        components = {name: scores[i] for (name, _), scores in zip(weights, per_reward, strict=True)}
        components["weighted"] = sum(weight * components[name] for name, weight in weights)
        rewards.append(components)
    return rewards
