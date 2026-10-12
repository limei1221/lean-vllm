"""The registry's exposition format, and what the engine actually records."""

import pytest
from conftest import by_label, observed, total

from lean_vllm.engine.metrics import Metrics
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class TestExposition:
    def test_a_counter_renders_its_total(self):
        metrics = Metrics()
        metrics.requests_received.inc()
        metrics.requests_received.inc(3)
        assert "lean_vllm:num_requests_received_total 4.0" in metrics.render().splitlines()

    def test_a_labelled_counter_renders_one_line_per_value(self):
        metrics = Metrics()
        for reason in ("stop", "length", "stop"):
            metrics.requests_finished.labels(reason).inc()
        lines = metrics.render().splitlines()
        assert 'lean_vllm:request_success_total{finished_reason="length"} 1.0' in lines
        assert 'lean_vllm:request_success_total{finished_reason="stop"} 2.0' in lines

    def test_a_gauge_renders_the_last_value_set(self):
        metrics = Metrics()
        metrics.running.set(5)
        metrics.running.set(2)
        assert "lean_vllm:num_requests_running 2.0" in metrics.render().splitlines()

    def test_histogram_buckets_are_cumulative(self):
        metrics = Metrics()
        for value in (0.005, 0.5, 100.0):
            metrics.e2e.observe(value)
        lines = metrics.render().splitlines()
        name = "lean_vllm:e2e_request_latency_seconds"
        assert f'{name}_bucket{{le="0.01"}} 1.0' in lines
        assert f'{name}_bucket{{le="1.0"}} 2.0' in lines
        assert f'{name}_bucket{{le="+Inf"}} 3.0' in lines
        assert f"{name}_count 3.0" in lines and f"{name}_sum 100.505" in lines

    def test_two_engines_count_apart(self):
        first, second = Metrics(), Metrics()
        first.requests_received.inc()
        assert "lean_vllm:num_requests_received_total 0.0" in second.render().splitlines()

    def test_every_metric_is_declared_before_it_is_sampled(self):
        """A sample with no preceding # TYPE is not scrapeable."""
        declared = set()
        for line in Metrics().render().splitlines():
            if line.startswith("# TYPE"):
                declared.add(line.split(" ")[2])
            elif not line.startswith("#"):
                name = line.split(" ")[0].split("{")[0]
                base = name.rsplit("_", 1)[0] if name.endswith(("_bucket", "_sum", "_count")) else name
                assert base in declared or name in declared, name


class TestEngineRecording:
    def test_a_finished_request_lands_in_the_latency_histograms(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert observed(metrics.ttft) == 1 and observed(metrics.e2e) == 1
        assert observed(metrics.queue_time) == 1
        assert observed(metrics.tpot) == 1  # three tokens, so there is a per-token rate
        assert by_label(metrics.requests_finished) == {"length": 1}

    def test_a_one_token_request_records_no_tpot(self, make_engine):
        """TPOT is the rate after the first token, and one token has no after."""
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=1, ignore_eos=True))
        engine.run_to_completion()
        assert observed(engine.metrics.ttft) == 1
        assert observed(engine.metrics.tpot) == 0

    def test_token_counters_split_prefill_from_decode(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert total(metrics.prefill_tokens) == 8  # one chunk of the whole prompt
        assert total(metrics.decode_tokens) == 2  # the prefill step sampled the first token
        assert total(metrics.prompt_tokens) == 8 and total(metrics.generation_tokens) == 3

    def test_queue_depths_and_kv_usage_are_gauged(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert total(engine.metrics.running) == 2
        assert total(engine.metrics.waiting) == 0
        assert total(engine.metrics.kv_usage) == 1.0  # both blocks taken

    def test_preemptions_are_counted(self, make_engine):
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8, max_num_seqs=2)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        for _ in range(4):
            engine.step()
        assert total(engine.metrics.preemptions) > 0

    def test_the_prefix_cache_hit_rate_counts_blocks(self, make_engine):
        engine = make_engine()
        engine.add(prompt(16), FOREVER)
        engine.step()
        engine.add(prompt(16), FOREVER)  # same prompt, so one full block hits
        engine.step()
        metrics = engine.metrics
        assert total(metrics.prefix_cache_queries) == 4  # two blocks each
        assert total(metrics.prefix_cache_hits) == 1  # the trailing block is never a candidate

    def test_admitted_requests_are_counted(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        assert total(engine.metrics.requests_received) == 2

    def test_a_step_that_scheduled_nothing_is_not_a_forward_pass(self, make_engine):
        """Otherwise an idle poll loop would inflate the step count and deflate the busy fraction."""
        engine = make_engine()
        engine.step()
        assert total(engine.metrics.steps) == 0
        engine.add(prompt(8), FOREVER)
        engine.step()
        assert total(engine.metrics.steps) == 1

    def test_each_step_is_counted_by_how_it_ran(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert by_label(metrics.requests_finished) == {"length": 1}
        assert total(metrics.graph_steps) == 0  # the fake runner captures no graphs
        assert by_label(metrics.eager_steps) == {"enforced": total(metrics.steps)}
        # the same clock as model_busy, split by kind
        assert by_label(metrics.step_seconds)["enforced"] == pytest.approx(total(metrics.model_busy))
        assert observed(metrics.e2e) == 1
