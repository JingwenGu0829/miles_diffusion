"""Pluggable API reward actors for externally managed services."""

from __future__ import annotations

import base64
import io
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real

import torch
from PIL import Image

from miles.utils.misc import SingletonMeta, load_function
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ApiRewardActor(ABC):
    """Base for API reward actors using externally managed services."""

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        if len(outputs) != len(prompts):
            raise ValueError("API reward requires one prompt per output")
        if not outputs:
            return []
        scores = self._score_batch(outputs, prompts)
        if len(scores) != len(outputs):
            raise ValueError("API reward actor must return one score per output")
        if any(isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) for score in scores):
            raise ValueError("API reward scores must be finite numbers")
        return [float(score) for score in scores]

    @abstractmethod
    def _score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        """Return one score per CFHW tensor in input order; calls may run concurrently."""
        raise NotImplementedError


class AsyncApiRewardPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """API reward pool with one zero-GPU actor handling concurrent HTTP requests."""

    name = "api"
    actor_base_cls = ApiRewardActor

    def __init__(self, args) -> None:
        config = args._api_rm_config
        if config is None:
            raise ValueError("API reward requires --api-rm-config.")
        actor_cls = load_function(config.actor_class)
        if not isinstance(actor_cls, type) or not issubclass(actor_cls, self.actor_base_cls):
            raise TypeError(f"API reward actor_class must be an {self.actor_base_cls.__name__} subclass")
        super().__init__(
            actor_cls=actor_cls,
            actor_kwargs=config.actor_kwargs,
            num_workers=1,
            batch_size=1,
            num_gpus_per_worker=0,
            colocate=False,
            name=self.name,
            actor_max_concurrency=config.max_concurrency,
        )


async def api_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    pool = AsyncApiRewardPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "api", max_queue_depth)
    return scores
