# Tiered long-context qualification

Batch-one qualification on JUPITER allocation `958529`, using the compiled
hot/cold stream-overlap implementation, the exact 400K HBM MLA cache, and
otherwise the same TP4/EP4 configuration as the preceding overlap result.

## High-water reserve

The 5 GB planned HBM reserve was sufficient at startup but not after a request.
Following two 4K runs, each GPU had about 1.5 GiB free. A 32K prefill then
failed when FlashMLA requested a 2.00 GiB sparse-attention buffer with only
1.68 GiB physically free.

Increasing `--tiered-moe-hbm-reserve-gb` from 5 to 7 moved 103 additional
layer-expert slots per rank to pinned Grace memory. The resulting placement is
3,176 hot and 1,624 cold slots per rank. It started with 6.29 GiB free HBM and
retained 2.54 GiB after the 32K request.

| Input / output | TTFT | TPOT | P99 ITL | Decode tok/s | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32,768 / 256 | 7.729 s | 27.60 ms | 28.29 ms | 36.24 | Pass |
| 399,744 / 256 | 118.385 s | 27.35 ms | 28.97 ms | 36.57 | Pass |

## Exact-capacity correction

The first 399,744 + 256 run reached its first token in about 118.2 seconds,
then paused after 193 output tokens. That boundary is diagnostic: vLLM's block
pool permanently removes one physical KV block for its null block. The tiered
planner had allocated 6,250 physical blocks, leaving only 6,249 usable; token
399,937 needed the missing block.

The planner now includes the scheduler's one null block. The corrected plan has
6,251 physical blocks, or 400,064 physical token slots and 400,000 usable token
slots. This adds 3,452,160 bytes per rank and does not change expert placement.
Ruff check, Ruff format, and all 29 focused tiered tests pass.

The native references are 7.652 seconds TTFT and 37.06 decode tok/s at 32K,
and 120.853 seconds TTFT and 37.57 decode tok/s at 399,744 tokens.
The corrected tiered run is therefore about 2.47 seconds faster to first token
and 2.7% slower in decode at full context. It finished all 400,000 input plus
output tokens with 2.52 GiB physically free per GPU.
