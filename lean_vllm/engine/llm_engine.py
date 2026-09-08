import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from lean_vllm.config import Config
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.sequence import Sequence
from lean_vllm.engine.metrics import Metrics
from lean_vllm.engine.scheduler import QueueFull, Scheduler
from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.utils.detokenizer import IncrementalDetokenizer


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.metrics = Metrics()
        self.detokenizers: dict[str, IncrementalDetokenizer] = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, request_id: str | None = None) -> str:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, request_id)
        detokenizer = IncrementalDetokenizer(self.tokenizer, prompt, seq.skip_special_tokens)
        try:
            self.scheduler.add(seq)    # last, so a refused request leaves nothing behind
        except QueueFull:
            self.metrics.record_rejected()
            raise
        self.metrics.record_received()
        self.detokenizers[seq.request_id] = detokenizer
        return seq.request_id

    def abort_request(self, request_id: str) -> bool:
        aborted = self.scheduler.abort(request_id)
        self.detokenizers.pop(request_id, None)
        if aborted:
            self.metrics.record_aborted()
        return aborted

    def step(self) -> tuple[list[RequestOutput], int, int]:
        started = perf_counter()
        output = self.scheduler.schedule()
        stepped = []
        if output:
            token_ids = self.model_runner.call("run", output.scheduled)
            stepped = self.scheduler.postprocess(output.scheduled, token_ids)
        outputs = [self._output(seq) for seq in stepped] + [self._dropped(seq) for seq in output.dropped]
        self.metrics.record_step(
            self.scheduler, output, outputs, perf_counter() - started, self.model_runner.eager_reason
        )
        return outputs, output.num_prefill_tokens, output.num_decode_tokens

    def _output(self, seq: Sequence) -> RequestOutput:
        token_id = seq.last_token
        detokenizer = self.detokenizers[seq.request_id]
        text = detokenizer.decode(token_id)
        if seq.is_finished:
            del self.detokenizers[seq.request_id]
        return RequestOutput(
            request_id=seq.request_id,
            token_ids=[token_id],
            text=text,
            finished=seq.is_finished,
            finish_reason=seq.finish_reason,
            metrics=seq.metrics() if seq.is_finished else None,
        )

    def _dropped(self, seq: Sequence) -> RequestOutput:
        """Finished by the scheduler without ever sampling, so there is no token."""
        self.detokenizers.pop(seq.request_id, None)
        return RequestOutput(
            request_id=seq.request_id,
            token_ids=[],
            finished=True,
            finish_reason=seq.finish_reason,
            metrics=seq.metrics(),
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        request_ids = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        collected = {request_id: {"text": "", "token_ids": []} for request_id in request_ids}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            step_outputs, num_prefill_tokens, num_decode_tokens = self.step()
            elapsed = perf_counter() - t
            if num_prefill_tokens:
                prefill_throughput = num_prefill_tokens / elapsed
            if num_decode_tokens:
                decode_throughput = num_decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for output in step_outputs:
                collected[output.request_id]["text"] += output.text
                collected[output.request_id]["token_ids"] += output.token_ids
                if output.finished:
                    pbar.update(1)
        pbar.close()
        return [collected[request_id] for request_id in request_ids]
