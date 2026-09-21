"""Reward names shared by CLI dispatch and weighted mixtures."""

from collections.abc import Mapping
from functools import partial

from miles.utils.misc import load_function


async def _call_builtin(path, args, samples):
    return await load_function(path)(args, samples)


_BUILTIN_REWARDS = {
    "hps": partial(_call_builtin, "miles.rollout.rm_hub.hps.hps_rm"),
    "pickscore": partial(_call_builtin, "miles.rollout.rm_hub.pickscore.pickscore_rm"),
    "ocr": partial(_call_builtin, "miles.rollout.rm_hub.ocr.ocr_rm"),
    "api": partial(_call_builtin, "miles.rollout.rm_hub.api.api_rm"),
    "openai_api": partial(_call_builtin, "miles.rollout.rm_hub.openai_api.openai_api_rm"),
}


def get_reward_registry(args=None) -> dict:
    rewards = dict(_BUILTIN_REWARDS)
    if path := getattr(args, "custom_rm_registry_path", None):
        custom_rewards = load_function(path)
        if not isinstance(custom_rewards, Mapping):
            raise TypeError("--custom-rm-registry-path must point to a mapping of names to reward callables")
        for name, reward in custom_rewards.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip() or any(c in name for c in ",="):
                raise ValueError(f"Invalid custom reward name: {name!r}")
            if name in rewards or name == "weighted":
                raise ValueError(f"Custom reward name {name!r} is reserved")
            if not callable(reward):
                raise TypeError(f"Custom reward {name!r} must be an async batched reward callable")
            rewards[name] = reward
    return rewards
