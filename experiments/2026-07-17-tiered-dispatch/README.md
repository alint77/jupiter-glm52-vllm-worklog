# Tiered Marlin dispatch

First Phase 3 execution slice on JUPITER allocation `957083`.

Each routed layer builds one Marlin backend for its compact hot tensors and one
for its compact cold CUDA aliases. The primary backend is registered as vLLM's
internal modular kernel, so native sequence-parallel prepare/dispatch runs once.
Both tiers consume the same prepared top-8 routing result with disjoint
global-to-local maps. Their rank-local outputs are added before the primary
kernel performs one native finalize/combine. Shared experts remain on vLLM's
existing path.

The first implementation left the primary kernel unregistered. vLLM therefore
ran its legacy outer dispatch and the synthetic kernel tried to dispatch the
already gathered tokens again. The four-rank profile failed the communicator's
token-size assertion. Registering the primary internal kernel removed the
legacy path.

An instrumented 8,192-token profile then completed on all four ranks:

```text
12:06:32  prepare start/complete
12:06:32  hot tier start: 60 experts
12:06:33  hot tier complete; cold tier start: 4 experts
12:06:33  cold tier complete; finalize start/complete
```

This is a coarse first-layer trace, not a benchmark: it establishes that one
prepare, both HBM/UVA Marlin calls, the join, and one finalize all execute.
Focused tests enforce the one-prepare/one-finalize contract and quant-method
wiring; Ruff and Python 3.12 mypy pass.

Startup next reached KV capacity planning and reported `-5.07 GiB` available.
That is the expected next implementation boundary: ordinary vLLM still prices
and allocates the main MLA cache as HBM even though this plan selected
`host_uva`. Output comparison, stream overlap, compile, and CUDA-graph work are
not yet claimed.
