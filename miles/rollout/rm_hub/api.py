"""Image rewards over the OpenAI-compatible Chat Completions API.

``--api-rm-config`` is a YAML mapping of reward names to configurations, e.g.::

    gemini:
      model: your-vision-model
      base_url: https://generativelanguage.googleapis.com/v1beta/openai/
      api_key_env: GEMINI_API_KEY

Use ``--rm-type gemini`` or include ``gemini=0.3`` in the weighted-mixture example.
Optional fields: prompt_path (relative to this YAML), score_min, score_max,
timeout_s, and max_concurrency. Requests are never retried or replaced by zero.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from weakref import WeakKeyDictionary

from miles.utils.types import Sample

from .api_utils import ApiRewardClient, get_api_rm_configs


# Each loop owns its clients and semaphores. The rollout manager reuses one loop
# across microgroups/rollouts; tests or custom callers can use independent loops.
_clients: WeakKeyDictionary = WeakKeyDictionary()


async def close_api_rm_clients() -> None:
    clients = _clients.pop(asyncio.get_running_loop(), {})
    await asyncio.gather(*(client.client.close() for client in clients.values()))


async def api_rm(args, samples: Sequence[Sample], *, name: str | None = None, **kwargs) -> list[float]:
    name = name or args.rm_type
    configs = get_api_rm_configs(args)
    if name not in configs:
        raise ValueError(f"API reward {name!r} is not configured in --api-rm-config")
    config = configs[name]
    clients = _clients.setdefault(asyncio.get_running_loop(), {})
    key = (name, config)
    if key not in clients:
        clients[key] = ApiRewardClient(name, config)
    # gather preserves input order, regardless of HTTP completion order. On
    # failure cancel sibling requests, then propagate rather than return a subset.
    tasks = [asyncio.create_task(clients[key].score_one(sample)) for sample in samples]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
