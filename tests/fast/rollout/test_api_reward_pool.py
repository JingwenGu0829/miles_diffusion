"""Real zero-GPU Ray actors against a local HTTP judge; no provider credentials needed."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45, suite="stage-a-cpu", labels=[])

import asyncio
import json
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import ray
import torch

import miles.rollout.rm_hub.api as api_module
import miles.rollout.rm_hub.hps as hps_module
from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.rm_hub.api import ApiRewardConfig, api_rm, close_api_rm_pools
from miles.rollout.rm_hub.weighted_mixture_rm import weighted_mixture_rm
from miles.utils.types import Sample


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(
        address="local",
        num_cpus=0,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=80 * 1024 * 1024,
        runtime_env={"env_vars": {"TEST_RM_KEY": "test-only-secret", "NO_PROXY": "127.0.0.1"}},
    )
    yield
    ray.shutdown()


@pytest_asyncio.fixture(autouse=True)
async def cleanup_pools(ray_cluster):
    yield
    await close_api_rm_pools()


@pytest.fixture
def judge():
    lock = threading.Lock()
    state = Namespace(requests=[], completed=[], active=0, peak=0, score=lambda payload: 1.0)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state.requests.append(payload)
                state.active += 1
                state.peak = max(state.peak, state.active)
            try:
                assert self.headers["Authorization"] == "Bearer test-only-secret"
                score = state.score(payload)
                status = 200
                result = {
                    "id": "local-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps({"score": score})},
                        }
                    ],
                }
            except Exception as exc:
                status = 500
                result = {"error": {"message": str(exc)}}
            finally:
                with lock:
                    state.active -= 1
                    state.completed.append(payload)
            body = json.dumps(result).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Failure tests intentionally terminate the requesting actor.

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def _args(server, **configs):
    return Namespace(
        rm_type=next(iter(configs)),
        custom_rm_path=None,
        _api_rm_configs={
            name: ApiRewardConfig(model=name, base_url=server.url, api_key_env="TEST_RM_KEY", timeout_s=15, **config)
            for name, config in configs.items()
        },
    )


def _samples(*indices):
    return [Sample(index=i, prompt=str(i), generated_output=torch.full((3, 1, 8, 8), i / 4)) for i in indices]


def _index(payload):
    return int(payload["messages"][1]["content"][0]["text"])


async def _assert_actors_dead(actors):
    async def wait_dead(actor):
        # ray.kill returns before the actor process has necessarily exited.
        while True:
            try:
                await asyncio.wrap_future(actor.close.remote().future())
            except ray.exceptions.RayActorError:
                return
            await asyncio.sleep(0.01)

    await asyncio.wait_for(asyncio.gather(*(wait_dead(actor) for actor in actors)), timeout=10)


async def test_shared_concurrency_order_and_normal_shutdown(judge):
    first_pair = threading.Barrier(2, timeout=15)

    def score(payload):
        index = _index(payload)
        if index in (0, 1):
            first_pair.wait()
        if index == 0:
            time.sleep(0.15)
        return index

    judge.score = score
    args = _args(judge, judge={"max_concurrency": 2})
    first, second = _samples(0, 1), _samples(2, 3)
    rewards = await asyncio.wait_for(asyncio.gather(api_rm(args, first), api_rm(args, second)), timeout=60)
    assert rewards == [[0.0, 1.0], [2.0, 3.0]]
    assert _index(judge.completed[0]) == 1
    assert judge.peak == 2
    assert second[0].reward_max_queue_depth["judge"] >= 1
    actors = list(api_module._pools["judge"]._actors)
    await close_api_rm_pools()
    await _assert_actors_dead(actors)


async def test_local_and_two_api_rewards_share_dispatch_without_alias_crosstalk(judge, monkeypatch):
    def score(payload):
        index = _index(payload)
        return index if payload["model"] == "judge" else 4 - index

    judge.score = score
    monkeypatch.setattr(hps_module, "hps_rm", AsyncMock(return_value=[0.1, 0.2]))
    args = _args(judge, judge={"max_concurrency": 1}, reverse={"max_concurrency": 1})
    args.custom_rm_args = "hps=0.2,judge=0.5,reverse=0.3"
    args.reward_key = "weighted"
    rewards = await asyncio.wait_for(weighted_mixture_rm(args, _samples(1, 2)), timeout=60)
    assert [r["weighted"] for r in rewards] == pytest.approx([1.42, 1.64])
    assert [(r["judge"], r["reverse"]) for r in rewards] == [(1, 3), (2, 2)]
    assert set(api_module._pools) == {"judge", "reverse"}
    samples = _samples(1, 2)
    samples[1].metadata = {"rm_type": "reverse"}
    assert await batched_async_rm(args, samples) == [1.0, 2.0]


@pytest.mark.parametrize("failure", ["http", "cancel"])
async def test_failure_or_cancellation_terminates_actors_and_discards_queue(judge, failure):
    started = threading.Event()
    release = threading.Event()

    def score(payload):
        started.set()
        if not release.wait(20):
            raise TimeoutError("Test did not release HTTP request")
        if failure == "http":
            raise RuntimeError("judge unavailable")
        return 1

    judge.score = score
    args = _args(judge, judge={"max_concurrency": 1})
    # Warm the actor before exercising the failure path.
    judge.score = lambda payload: 1
    await asyncio.wait_for(api_rm(args, _samples(0)), timeout=60)
    judge.requests.clear()
    judge.score = score
    actors = list(api_module._pools["judge"]._actors)
    task = asyncio.create_task(api_rm(args, _samples(1, 2, 3)))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        if failure == "http":
            release.set()
            with pytest.raises(RuntimeError, match="judge unavailable"):
                await asyncio.wait_for(task, timeout=10)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await _assert_actors_dead(actors)
        with pytest.raises(RuntimeError, match="pool is closed"):
            await api_rm(args, _samples(4))
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    # Requests already running on the provider may finish, but queued work must not drain.
    assert len(judge.requests) == 1
