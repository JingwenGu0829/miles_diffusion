"""Pluggable API reward actors for externally managed services."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real

import torch
from PIL import Image

from miles.utils.api_rm_config import OpenAIImageRewardConfig
from miles.utils.misc import SingletonMeta, load_function
from miles.utils.processing_utils import generated_output_to_rgb_hwc_uint8_frames
from miles.utils.types import Sample

from .core import AsyncRewardActorPool, record_reward_queue_depth


class ApiRewardActor(ABC):
    """Base for remote API rewards using the shared Ray pool.

    Implement ``_score_batch`` to encode raw CFHW tensors, call the service, and
    return one numeric score per input in the same order. Authentication, media
    encoding, HTTP schemas, and score ranges belong to the implementation.
    The pool may call the actor concurrently; keep request state local to each call.
    """

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
        raise NotImplementedError


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_reward",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_score(content: str, config: OpenAIImageRewardConfig) -> float:
    score = json.loads(content)["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Reward score must be a finite number")
    if not config.score_min <= score <= config.score_max:
        raise ValueError(f"Reward score must be in [{config.score_min}, {config.score_max}]")
    return float(score)


class OpenAIImageScorer:
    """Score prompt/image pairs using the OpenAI-compatible Chat Completions API."""

    def __init__(self, config: OpenAIImageRewardConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=2,
        )

    def __call__(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        scores = []
        for prompt, image in zip(prompts, images, strict=True):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                        ],
                    },
                ],
                response_format=_RESPONSE_FORMAT,
            )
            scores.append(_parse_score(response.choices[0].message.content, self.config))
        return scores


class OpenAIImageRewardActor(ApiRewardActor):
    def __init__(self, **kwargs) -> None:
        self.scorer = OpenAIImageScorer(OpenAIImageRewardConfig(**kwargs))

    def _score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        images = []
        for output in outputs:
            (frame,) = generated_output_to_rgb_hwc_uint8_frames(output, None, round_normalized=True)
            images.append(Image.fromarray(frame))
        return self.scorer(prompts, images)


class AsyncApiRewardPool(AsyncRewardActorPool, metaclass=SingletonMeta):
    """API reward pool with one zero-GPU actor handling concurrent HTTP requests."""

    def __init__(self, args) -> None:
        config = args._api_rm_config
        if config is None:
            raise ValueError("API reward requires --api-rm-config.")
        actor_cls = load_function(config.actor_class)
        if not isinstance(actor_cls, type) or not issubclass(actor_cls, ApiRewardActor):
            raise TypeError("API reward actor_class must be an ApiRewardActor subclass")
        super().__init__(
            actor_cls=actor_cls,
            actor_kwargs=config.actor_kwargs,
            num_workers=1,
            batch_size=1,
            num_gpus_per_worker=0,
            colocate=False,
            name="api",
            actor_max_concurrency=config.max_concurrency,
        )


async def api_rm(args, samples: Sequence[Sample], **kwargs) -> list[float]:
    pool = AsyncApiRewardPool(args)
    scores, max_queue_depth = await pool.score([s.generated_output for s in samples], [s.prompt for s in samples])
    record_reward_queue_depth(samples, "api", max_queue_depth)
    return scores
