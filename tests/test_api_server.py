"""The HTTP layer against an in-process fake engine, so no model is needed."""

import json

import pytest

pytest.importorskip("fastapi", reason="the serve extra is not installed")

from fastapi.testclient import TestClient

from lean_vllm.engine.async_engine import EngineDeadError
from lean_vllm.engine.metrics import Metrics
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import InvalidRequest, QueueFull
from lean_vllm.entrypoints.api_server import build_app

MODEL = "fake-model"


class FakeTokenizer:
    """One id per character, so a prompt's token count is its length."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, return_dict=True):
        ids = self.encode("\n".join(message["content"] for message in messages))
        return {"input_ids": ids} if return_dict else ids    # as a real fast tokenizer does


class FakeAsyncEngine:
    """Scripted outputs, so what is under test is the HTTP layer and nothing else."""

    def __init__(self, pieces=("Hello", ", world"), finish_reason="length", max_model_len=64):
        self.tokenizer = FakeTokenizer()
        self.max_model_len = max_model_len
        self.metrics = Metrics()
        self.pieces = list(pieces)
        self.finish_reason = finish_reason
        self.is_dead = False
        self.error = None
        self.admission_error: Exception | None = None
        self.requests: list[tuple] = []
        self.aborted: list[str] = []

    def start(self):
        pass

    def stop(self):
        pass

    async def add_request(self, prompt, sampling_params, request_id=None):
        self.requests.append((prompt, sampling_params, request_id))
        if self.admission_error is not None:
            self.metrics.record_rejected()
            raise self.admission_error
        self.metrics.record_received()
        return self._outputs(request_id)

    async def _outputs(self, request_id):
        try:
            for i, piece in enumerate(self.pieces):
                last = i == len(self.pieces) - 1
                yield RequestOutput(
                    request_id=request_id,
                    token_ids=[i],
                    text=piece,
                    finished=last,
                    finish_reason=self.finish_reason if last else None,
                )
        finally:
            self.aborted.append(request_id)

    def abort(self, request_id):
        self.aborted.append(request_id)


@pytest.fixture
def engine():
    return FakeAsyncEngine()


@pytest.fixture
def client(engine):
    with TestClient(build_app(engine, MODEL)) as client:
        yield client


def complete(client, **overrides) -> dict:
    body = {"model": MODEL, "prompt": "hi", "max_tokens": 4} | overrides
    return client.post("/v1/completions", json=body)


def events(response) -> list[str]:
    return [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]


class TestEndpoints:

    def test_health_is_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_health_is_503_once_the_engine_is_dead(self, client, engine):
        engine.is_dead, engine.error = True, RuntimeError("boom")
        assert client.get("/health").status_code == 503

    def test_models_lists_the_served_name(self, client):
        assert [card["id"] for card in client.get("/v1/models").json()["data"]] == [MODEL]

    def test_metrics_is_prometheus_text(self, client):
        complete(client)
        response = client.get("/metrics")
        assert response.headers["content-type"].startswith("text/plain")
        assert "# TYPE lean_vllm:num_requests_received_total counter" in response.text
        assert "lean_vllm:num_requests_received_total 1" in response.text

    def test_metrics_json_is_the_benchmark_summary(self, client):
        complete(client)
        summary = client.get("/metrics.json").json()
        assert summary["requests"]["received"] == 1
        assert summary["prefix_cache_hit_rate"] is None    # nothing scheduled behind this fake


class TestCompletions:

    def test_the_pieces_are_joined(self, client):
        body = complete(client).json()
        assert body["choices"][0]["text"] == "Hello, world"
        assert body["choices"][0]["finish_reason"] == "length"

    def test_usage_counts_both_ends(self, client):
        assert complete(client).json()["usage"] == {
            "prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4
        }

    def test_token_ids_are_accepted_as_a_prompt(self, client, engine):
        complete(client, prompt=[1, 2, 3])
        assert engine.requests[0][0] == [1, 2, 3]

    def test_temperature_zero_is_passed_through_as_greedy(self, client, engine):
        complete(client, temperature=0)
        assert engine.requests[0][1].temperature == 0

    def test_the_priority_extra_reaches_the_sampling_params(self, client, engine):
        complete(client, priority=3, ignore_eos=True)
        assert (engine.requests[0][1].priority, engine.requests[0][1].ignore_eos) == (3, True)


class TestChatCompletions:

    def test_the_chat_template_builds_the_prompt(self, client, engine):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        response = client.post("/v1/chat/completions", json=body)
        assert response.json()["choices"][0]["message"] == {"role": "assistant", "content": "Hello, world"}
        assert engine.requests[0][0] == [ord("h"), ord("i")]


class TestStreaming:

    def test_deltas_are_sse_and_end_with_done(self, client):
        response = complete(client, stream=True)
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = events(response)
        assert payloads[-1] == "[DONE]"
        assert "".join(json.loads(p)["choices"][0]["text"] for p in payloads[:-1]) == "Hello, world"
        assert json.loads(payloads[-2])["choices"][0]["finish_reason"] == "length"

    def test_a_chat_stream_opens_with_the_role(self, client):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True}
        payloads = events(client.post("/v1/chat/completions", json=body))
        assert json.loads(payloads[0])["choices"][0]["delta"] == {"role": "assistant", "content": ""}

    def test_include_usage_adds_a_final_choiceless_chunk(self, client):
        payloads = events(complete(client, stream=True, stream_options={"include_usage": True}))
        usage_chunk = json.loads(payloads[-2])
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"]["completion_tokens"] == 2

    def test_usage_is_absent_unless_asked_for(self, client):
        assert all("usage" not in json.loads(p) for p in events(complete(client, stream=True))[:-1])


class TestStopStrings:

    def test_a_stop_string_truncates_the_text_and_aborts(self, client, engine):
        body = complete(client, stop=",").json()
        assert body["choices"][0]["text"] == "Hello"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert engine.aborted == [body["id"]]    # the request is dropped, not run to max_tokens


class TestRefusals:

    @pytest.mark.parametrize("field, value", [
        ("top_p", 0.9), ("top_k", 20), ("seed", 1), ("logprobs", 1),
        ("presence_penalty", 0.1), ("logit_bias", {"1": 1.0}), ("echo", True), ("best_of", 2),
    ])
    def test_an_unsupported_parameter_is_named_in_a_400(self, client, field, value):
        response = complete(client, **{field: value})
        assert response.status_code == 400
        assert field in response.json()["error"]["message"]

    def test_an_openai_default_a_client_never_set_is_not_punished(self, client):
        """`"top_p": 1.0` asks for nothing, and clients send it anyway."""
        assert complete(client, top_p=1.0, presence_penalty=0, echo=False, seed=None).status_code == 200

    def test_n_greater_than_one_is_refused(self, client):
        response = complete(client, n=2)
        assert response.status_code == 400
        assert "n > 1" in response.json()["error"]["message"]

    def test_an_unknown_field_is_refused_rather_than_ignored(self, client):
        assert complete(client, nucleus_sampling=True).status_code == 400

    def test_a_model_the_server_does_not_serve_is_a_404(self, client, engine):
        response = complete(client, model="some-other-model")
        assert response.status_code == 404
        assert "does not exist" in response.json()["error"]["message"]
        assert not engine.requests

    def test_the_served_name_is_the_one_that_works(self, client):
        assert complete(client, model=MODEL).status_code == 200
        assert client.post("/v1/chat/completions", json={
            "model": "some-other-model", "messages": [{"role": "user", "content": "hi"}],
        }).status_code == 404

    def test_a_prompt_over_the_context_is_refused(self, client, engine):
        response = complete(client, prompt=[0] * (engine.max_model_len + 1))
        assert response.status_code == 400
        assert not engine.requests    # refused here, not asserted deep in the runner

    def test_max_tokens_that_overruns_the_context_is_refused(self, client, engine):
        assert complete(client, prompt=[0] * 60, max_tokens=10).status_code == 400

    def test_a_prompt_the_engine_rejects_is_a_400_and_not_a_500(self, client, engine):
        """Token ids are only checkable against the vocabulary, which lives in the engine."""
        engine.admission_error = InvalidRequest("token id 999999 is outside the 100-token vocabulary")
        response = complete(client, prompt=[999999])
        assert response.status_code == 400
        assert "outside the 100-token vocabulary" in response.json()["error"]["message"]

    def test_a_full_queue_is_a_429(self, client, engine):
        engine.admission_error = QueueFull("4 requests already waiting")
        assert complete(client).status_code == 429

    def test_a_dead_engine_is_a_503(self, client, engine):
        engine.is_dead, engine.error = True, RuntimeError("boom")
        assert complete(client).status_code == 503

    def test_an_engine_that_dies_during_admission_is_a_503(self, client, engine):
        engine.admission_error = EngineDeadError("the engine thread died")
        assert complete(client).status_code == 503

    def test_a_capacity_drop_is_a_503_rather_than_a_completion(self, client, engine):
        """"capacity" means the server could not serve it, so it is not an OpenAI finish reason."""
        engine.pieces, engine.finish_reason = [""], "capacity"
        response = complete(client)
        assert response.status_code == 503
        assert "capacity" in response.json()["error"]["message"]

    def test_a_timeout_drop_is_a_504(self, client, engine):
        """It waited too long to be served, which is not the same as being over capacity."""
        engine.pieces, engine.finish_reason = [""], "timeout"
        response = complete(client)
        assert response.status_code == 504
        assert "timeout" in response.json()["error"]["message"]

    def test_a_capacity_drop_mid_stream_rides_in_the_stream(self, client, engine):
        engine.pieces, engine.finish_reason = ["Hello", ""], "capacity"
        payloads = events(complete(client, stream=True))
        assert payloads[-1] == "[DONE]"
        assert json.loads(payloads[-2])["error"]["type"] == "server_error"
