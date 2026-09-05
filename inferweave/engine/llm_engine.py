import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from inferweave.config import Config
from inferweave.sampling_params import SamplingParams
from inferweave.engine.output import RequestOutput
from inferweave.engine.sequence import Sequence
from inferweave.engine.scheduler import Scheduler
from inferweave.engine.model_runner import ModelRunner
from inferweave.utils.detokenizer import IncrementalDetokenizer


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
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
        self.scheduler.add(seq)
        self.detokenizers[seq.request_id] = IncrementalDetokenizer(
            self.tokenizer, prompt, seq.skip_special_tokens
        )
        return seq.request_id

    def abort_request(self, request_id: str) -> bool:
        aborted = self.scheduler.abort(request_id)
        self.detokenizers.pop(request_id, None)
        return aborted

    def step(self) -> tuple[list[RequestOutput], int, int]:
        output = self.scheduler.schedule()
        if not output:
            return [], 0, 0
        token_ids = self.model_runner.call("run", output.scheduled)
        stepped = self.scheduler.postprocess(output.scheduled, token_ids)
        return [self._output(seq) for seq in stepped], output.num_prefill_tokens, output.num_decode_tokens

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
