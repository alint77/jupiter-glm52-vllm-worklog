# Narration script (draft 1)

One block per scene. Seconds = the scene's current length; the scenes will be
re-timed to the generated audio.

## S1 Hardware (~39 s)

This is one node of the JUPITER supercomputer: four NVIDIA Grace Hopper superchips. Each one pairs a 72-core Grace CPU and 120 gigabytes of LPDDR5X with a Hopper GPU and 96 gigabytes of HBM. The two are joined by NVLink C2C, at 450 gigabytes per second each way, so the GPU can read CPU memory directly. The four GPUs talk over NVLink, and nodes over InfiniBand. In practice we get about 3.6 terabytes per second from HBM, and about 420 gigabytes per second over C2C: the second-fastest way into the GPU.

## S2 The model doesn't fit (~26 s)

We want to serve MiMo V2.6 Pro: a trillion-parameter mixture-of-experts model, with 69 MoE layers of 384 experts each. Even in MXFP4 that's 566 gigabytes, and the node has 384 gigabytes of HBM. We could spread it over two nodes, but then every layer talks over InfiniBand. Or we stay on one node and keep part of the model in Grace memory.

## S3 Stock vLLM (~28 s)

vLLM can already offload. It moves whole layers to CPU memory, and every step streams all of their weights back over the link, including experts no token is routed to. HBM and the link take turns instead of working together, and most of the bytes moved are wasted. With speculative decoding, that gives around 70 to 80 tokens per second.

## S4 Layout (~40 s)

Here's our layout. Attention is split by heads across the four GPUs, and each GPU owns a quarter of every layer's experts: 96 of the 384. For one token, the router picks 8 experts per layer. But we decode with speculative decoding: a drafter proposes 7 tokens, and the model checks all 8 in a single step. Now around 49 experts are active per layer, up to 64. Every weight we read serves several tokens, and with more experts active, the load spreads more evenly.

## S4b Prefetch (~32 s)

The obvious way to offload is to prefetch: while layer N runs, copy the next layer's offloaded experts into a spare HBM buffer. In prefill, with thousands of tokens per step, each layer runs long enough to hide its copy, so offloading is free. In decode, with 8 tokens, a layer takes about a quarter of a millisecond, but its copy takes about two. The GPU mostly waits. Copies only hide from roughly 256 tokens per step. So prefill prefetches, and decode needs something smarter.

## S5 Idea 1: read both memories at once (~77 s)

Decode is limited by memory bandwidth. So keep the frequently used, hot experts in HBM, put the cold ones in Grace memory, and read both at the same time. One after the other, a layer takes hot time plus cold time. Together, it takes only as long as the longer of the two. The ratio matters. At three hot experts to one cold, the cold read runs long and the GPU waits on it. At six to one, the cold read is fully hidden. At five to one, both finish together, because HBM is about five times faster than C2C. Now compare with reading all six experts from HBM alone: that takes longer. Offloading is actually faster than VRAM, because C2C adds its bandwidth on top. Even better, a single kernel can read both tiers: about twenty SMs stream cold experts, enough to saturate C2C, while the rest stream hot ones. That's 6 to 32 percent faster per layer than two kernels on two streams.

## S6 The problem (~20 s)

The catch: to fit the model, we have to offload far more than one expert in six. About 42 percent of the experts live in Grace memory. If every expert were used equally often, cold reads would take about four times as long as hot ones, and set the pace.

## S7 Idea 2: offload the experts nobody uses (~49 s)

But routing isn't uniform. Sort one layer's experts by how often they're used: a few take most of the traffic, and a long tail barely runs. So we profile on calibration data that looks like our deployment: agentic coding sessions and autoresearch ML tasks. Put the least-used half in Grace memory, and it receives only 18.5 percent of routes. Our production placement brings that down to 12.8 percent, well inside what the overlap can hide. How skewed routing is depends on the model, though: GLM 5.3 is flatter, and its least-used half still gets 23 percent.

## S8 Even then: the GPUs don't finish together (~32 s)

Even then, the GPUs don't finish together. Here's one layer on all four. Each GPU reads its hot and cold experts at the same time. GPUs one and two drew lots of cold experts, so cold reads set their pace. GPUs zero and three finish early, and sit idle in the collective for about 150 microseconds. Which GPU is last changes every layer: it's routing luck, not a slow GPU.

## S9 Idea 3: replicas (~50 s)

So, idea three: replicas. Back to the per-GPU layout: hot experts in HBM, cold ones in each GPU's Grace memory. In this step, 11 of the active experts are cold, and GPU three drew six of them, so every other GPU waits for it. But Grace memory has room to spare, so we keep copies of other GPUs' cold experts there. GPU three hands three of its cold reads to GPUs holding a copy, and the layer now waits for three instead of six. Every GPU sees the same routing, so all four make the same choice without talking to each other.

## S10 Putting it together (~21 s)

Putting it together: hot and cold experts read at once, in one kernel; placement profiled on deployment-like traffic; and replicas to balance the cold work. Prefill prefetches the next layer. On one node, that's around 210 tokens per second for a single user, against 70 to 80 with stock offloading.