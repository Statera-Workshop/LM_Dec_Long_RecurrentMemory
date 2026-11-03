# LM_Dec_Long_RecurrentMemory

RecurrentMemory - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Smoke-Tested**
- Failure class: Implementation Unverified: training diverged in three settings (default, detached memory, lr 1e-4)

## Provenance

- Source repository: (new) lucidrains/recurrent-memory-transformer-pytorch 0.7.0
- Upstream / license: lucidrains/recurrent-memory-transformer-pytorch (MIT), pip 0.7.0

## Differences from the source ([Unified] protocol)

- segments of 128 with 16 memory tokens and XL memories; gradients flow across segments (upstream wrapper can also do memory-replay backprop, not used)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.29 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.718 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 4.224 / test 4.247 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 1745 MB, inference 54 MB
- Peak GPU memory at length 8192 (batch 1): train 6875 MB, inference 57 MB

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy recurrent-memory-transformer-pytorch==0.7.0
python run.py check rmt
python run.py lm02 rmt      # enwik8 is downloaded to data/ on first use
```
