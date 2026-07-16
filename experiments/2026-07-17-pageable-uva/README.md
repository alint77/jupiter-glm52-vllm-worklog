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
