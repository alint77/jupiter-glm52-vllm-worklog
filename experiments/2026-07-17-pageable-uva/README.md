# Pageable Grace CUDA view

Phase 0 qualification of direct CUDA access to ordinary pageable Grace memory.

## Contract under test

- Input is a contiguous, non-pinned CPU tensor and an explicit CUDA device.
- The CUDA tensor aliases the same virtual address and keeps its CPU owner alive.
- The device must report pageable access through host page tables.
- Pinned and non-contiguous tensors are rejected.

## Measurements

The test job will record correctness and one-direction read bandwidth for HBM,
the existing pinned-UVA path, and the new pageable-UVA path. Results are pending
the editable vLLM build and clean native baseline run on Slurm job `956247`.
