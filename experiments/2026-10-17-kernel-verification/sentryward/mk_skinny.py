"""SentryWard input for skinny_kernel<64,128> (4 consumer warps + 1 TMA
producer, 8 stages). Usage: mk_skinny.py <grid> <K> <out.json>."""
import json, sys

M, N, TN, CK = 8, 128, 64, 128
grid, K = int(sys.argv[1]), int(sys.argv[2])
tiles, chunks = (N + TN - 1) // TN, (K + CK - 1) // CK
cfg = {
  "schema": "spverify.kernel-input/v1",
  "entry": "_ZN6skinny13skinny_kernelILi64ELi128EEEv14CUtensorMap_stS1_NS_6ParamsE",
  "launch": {"grid_dim": [grid, 1, 1], "block_dim": [160, 1, 1],
             "cluster_dim": [1, 1, 1], "dynamic_shared_bytes": 1024 + 8 * (TN * CK * 2 + 8 * CK * 2)},
  "declarations": [
    {"parameter": 2, "offset": 24, "type": "pointer", "space": "global", "object": "out"},
    {"parameter": 2, "offset": 32, "type": "pointer", "space": "global", "object": "ws"},
    {"parameter": 2, "offset": 40, "type": "pointer", "space": "global", "object": "counters"},
  ],
  "arguments": [{"parameter": 2, "offset": o, "type": "s32", "value": v}
                for o, v in [(0, M), (4, N), (8, K), (12, tiles), (16, chunks)]],
  "tma_descriptors": [
    {"parameter": 0, "offset": 0, "object": "w", "element_type": "bf16", "rank": 2,
     "global_shape": [K, N], "coordinate_strides_bytes": [2, 2 * K], "box_shape": [64, TN],
     "element_strides": [1, 1], "swizzle": "128B", "interleave": "none", "oob_fill": "zero"},
    {"parameter": 1, "offset": 0, "object": "x", "element_type": "bf16", "rank": 2,
     "global_shape": [K, M], "coordinate_strides_bytes": [2, 2 * K], "box_shape": [64, 8],
     "element_strides": [1, 1], "swizzle": "128B", "interleave": "none", "oob_fill": "zero"},
  ],
}
json.dump(cfg, open(sys.argv[3], "w"), indent=1)
