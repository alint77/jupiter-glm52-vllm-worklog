# Pageable Grace CUDA view

Phase 0 qualification of direct CUDA access to ordinary pageable Grace memory.

## Contract under test

- Input is a contiguous, non-pinned CPU tensor and an explicit CUDA device.
- The CUDA tensor aliases the same virtual address and keeps its CPU owner alive.
- The device must report pageable access through host page tables.
- Pinned and non-contiguous tensors are rejected.

## Measurements

Slurm job `956247`, GPU 0 and local Grace NUMA node 0:

- CUDA attributes 88 and 100 both report `1`.
- PyTorch `from_blob` rejects the unregistered pointer before launch.
- A DLPack-backed CUDA tensor bypasses that classification check without
  allocation, registration, or copying.
- The same virtual address is visible from CPU and GPU; bidirectional writes and
  owner lifetime pass. The complete UVA test file passes 4/4 tests.
- Triton's launcher separately rejects the unregistered pointer. PyTorch CUDA
  elementwise kernels accept it.

For a 1 GiB local allocation, the validated PyTorch read-plus-write probe reports:

| Source | Read bytes / time | Reported read bandwidth |
|---|---:|---:|
| HBM | 1 GiB / 0.596 ms | 1,801 GB/s |
| Existing pinned UVA | 1 GiB / 2.550 ms | 421 GB/s |
| Pageable CUDA alias | 1 GiB / 2.398 ms | 448 GB/s |

The important failure is residency: all 512 sampled pages begin on Grace NUMA
node 0 and all end on GPU-HBM NUMA node 4. CUDA preferred-host advice does not
stop this migration. Read-mostly advice also migrates all samples; OS `mlock`
still leaves half of the samples on node 4. Therefore this mechanism is a valid
same-address CUDA tensor, but not yet a valid persistent LPDDR tier.

Phase 0 decision: keep the implementation and tests as a capability probe, but
do not place cold experts or MLA cache in this allocation kind. The next
allocator experiment will use final-destination pinned UVA (already measured at
421 GB/s and proven at 40.33 GiB/rank by the native baseline) while retaining a
CPU fallback if registered-host allocation cannot satisfy the exact plan.

The first `GraceAllocation` contingency slice now allocates pinned CPU backing
first, creates its CUDA alias without another allocation, records exact bytes
and expected device/NUMA ownership, and supports a single direct copy into the
final backing. The final combined UVA and allocator suite passes 5/5 tests on
GPU 0; all changed-file pre-commit hooks also pass.

## Build verification

The complete editable source install succeeded with Torch 2.11.0+cu130 after a
66-minute cold build of vLLM's vendored FA2/FA3 and stable CUDA targets. Both
`vllm._C_stable_libtorch` and the new `vllm._pageable_grace_C` import through
the installed environment, and the registered pageable-view operator is
visible through `torch.ops._C`.

A second isolated `uv` install was stopped after it created a different
temporary Torch include path and began redundantly recompiling the full stable
kernel matrix. This was not a test or implementation failure. The targeted
package import passed, followed by a fresh GH200 run of the complete test file:
5/5 passed in 2.38 seconds.

The allocator now records a `GracePagePlacement` snapshot and optionally fails
closed when fewer than 95% of sampled pages reside on the expected Grace NUMA
node. A real 16 MiB pinned allocation passed strict auditing with 256/256 pages
on node 0 before and after a GPU write. The extended 5-test suite passes in
2.61 seconds, and the complete changed-file pre-commit hook set passes.
