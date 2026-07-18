# MTP2 control and profile

Job `969109` measured the unchanged tiered server with two draft tokens and a
size-3 verification graph. The batch-one request used 399,744 random input
tokens, 256 output tokens, greedy decoding, and seed 13.

| TTFT | TPOT | Decode | Acceptance | Accepted/drafted |
| ---: | ---: | ---: | ---: | ---: |
| 110.752 s | 8.757 ms | 114.20 tok/s | 75.74% | 153/202 |

Across eight profiled target steps and four ranks, routed Marlin occupied
7.413 ms per step on one stream. Each rank issued eight size-23 target-vocab
all-gathers and 16 size-16 draft-vocab all-gathers. This is the serial MTP2
control used by the overlap and local-argmax experiment.

The exact-400K continuation has SHA-256
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`,
matching the MTP3 and optimized runs. The four rank traces remain in
`/e/scratch/profound/naeimitabiei1/glm52-mtp2-profile-969109`.
