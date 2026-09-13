# Pipelined steps

Status: implemented, not yet measured. `LLMEngine.step` overlaps
detokenization and batch preparation with the forward pass instead of running
them serially after it.

## The idle it targets

A 12 September H100 offline trace measured the GPU idle 34.4% of the step
window, attributed to `detokenize` (1.72 ms/step) and `prepare_batch`
(1.23 ms/step) running serially with the forward pass — see
[the benchmark](../benchmarks/benchmark-2026-09-12.md). Neither phase depends
on that step's sampled output, so both can run while the GPU works on the next
step instead of blocking it.

## The loop

`step` launches a step and drains the one launched before it, so the drain's
host work runs in the shadow of the GPU work the launch just queued.
`Config.async_scheduling` picks which phases land in that shadow:

```
flag off                         flag on
--------                         -------
await tokens of step k-1         schedule step k
reconcile step k-1               launch step k
schedule step k                  await tokens of step k-1
launch step k                    reconcile step k-1
detokenize step k-1              detokenize step k-1
```

With the flag off, only detokenization moves. The drain still runs before
scheduling, so a stop condition is always seen before the next step is built,
and output is identical to before this change.

With the flag on, scheduling and batch preparation for step k also run ahead
of awaiting step k-1's tokens, so they land in the GPU's shadow too. That is
the prediction this change is meant to test against the baseline above; no
number has been measured for it yet.

## What the flag costs

A stop condition is seen one step late, so a step already launched for a
request that has just finished computes one extra token. That token is
discarded on reconcile and never emitted — output text and token ids are
unaffected — but the compute for it is spent. A request ending by hitting
`max_tokens` avoids this: the scheduler skips scheduling it a further step once
its reserved tokens already reach the limit. EOS and client stop sequences
still cost the one extra step.

`async_scheduling` is on by default, as it is in vLLM. With
`tensor_parallel_size` above 1 it turns itself off with a warning: ranks above
zero never see the sampled tokens, so they could not follow.

## Why awaiting a step doesn't wait for the next one

Tokens come back through `SampledTokens` (`lean_vllm/engine/sampled_tokens.py`),
which copies the sampled tensor to pinned host memory on its own CUDA stream
and records an event at launch time. Awaiting later waits on that event, not
on the default stream — which by the time the flag is on may already hold the
next step's kernels. A plain `.tolist()` on the device tensor would block on
the default stream instead, serializing the pipeline completely.

## What a reader needs to trust

- A pending token is a count, not a placeholder value — nothing ever reads a
  sampled token that doesn't exist yet.
- A row's pending token is discarded if its sequence was preempted or aborted
  since the step that would have produced it was launched.
- All device work stays on a single stream (aside from the token copy above).
  That is what makes it safe to free a KV block while a step is still in
  flight, and to publish a prefix-cache block whose KV a still-running kernel
  is writing: a later step's kernels run strictly after this step's writes,
  in stream order.

## What isn't measured yet

No numbers exist yet for the flag on vs. off. The run this predicts: an
offline GPU-idle-fraction trace at load 48 with the flag on and off, plus the
24–64 sweep against vLLM 0.26.0, against the baseline above.
