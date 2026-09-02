# CP correctness gates for the #52188 DFlash DCP port

Both run the **real** triton kernels on one GPU; no server, no allocation.
They extract `_prepare_dflash_inputs_kernel` from the working tree and from
git into standalone modules (triton requires `@jit` functions to live in real
files, hence the generated `knew.py` / `kold.py`).

```bash
.venv/bin/python gates/gate_cp1.py   # regression gate
.venv/bin/python gates/gate_cp4.py   # new-capability gate
```

- **gate_cp1.py** — at `cp_size=1` the ported kernel must be bit-identical to
  the pre-port kernel (`5c5dc1ac54`, i.e. commit B applied, C not yet), over
  all ten output tensors. 72 configs: block_size {16,64,128} x qlen {1,4,8,13}
  x 6 seeds. Block tables are seeded with zeros to exercise the null-block
  guard. **Result: 0 mismatches.**
- **gate_cp4.py** — at `cp_size>1` the ctx and query slots must match an
  independent torch reference of the DCP round-robin mapping
  (`block_table.py:283-300`). 72 configs: (cp_size, cp_interleave) in
  {(2,1),(4,1),(4,8),(8,4)} x every cp_rank x block_size {16,64} x qlen {4,8}.
  **Result: 0 mismatches.**

Note on test data: `num_rejected` must be drawn `< num_ctx`. A step verifies
`num_ctx` positions and accepts at least one, so rejections can never exceed
the context span; drawing them independently makes `num_valid_ctx` negative
and the kernel is not robust to that (nor does it need to be).
