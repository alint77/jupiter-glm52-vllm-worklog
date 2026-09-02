"""CP1 equivalence gate: the post-#52188 kernel must be bit-identical to the
pre-port kernel at cp_size=1. Runs both real triton kernels in one process."""
import sys, os
sys.path.insert(0, os.environ["CLAUDE_JOB_DIR"] + "/tmp")
import torch, triton
from knew import _prepare_dflash_inputs_kernel as new_k
from kold import _prepare_dflash_inputs_kernel as old_k

PAD_SLOT_ID = -1

dev = "cuda"
torch.manual_seed(0)
def run(kernel, is_new, *, num_reqs, qlen, nq, nsteps, block_size, max_num_reqs,
        max_num_tokens, max_model_len, bt_stride, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = num_reqs * qlen
    outs = dict(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32, device=dev),
        qpos=torch.zeros(max_num_tokens, dtype=torch.int64, device=dev),
        qsl=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=dev),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=dev),
        qslot=torch.zeros(max_num_tokens, dtype=torch.int64, device=dev),
        cpos=torch.zeros(total + 64, dtype=torch.int64, device=dev),
        cslot=torch.zeros(total + 64, dtype=torch.int64, device=dev),
        sidx=torch.zeros(max_num_reqs * nsteps, dtype=torch.int64, device=dev),
        spos=torch.zeros(max_num_reqs * nsteps, dtype=torch.int64, device=dev),
        sidxmap=torch.zeros(max_num_reqs * nsteps, dtype=torch.int32, device=dev),
    )
    base = torch.randint(1000, 5000, (num_reqs,), generator=g)
    tpos = torch.cat([base[i] + torch.arange(qlen) for i in range(num_reqs)]).to(dev).long()
    tqsl = (torch.arange(num_reqs + 1) * qlen).to(dev).int()
    idxmap = torch.arange(num_reqs, device=dev, dtype=torch.int32)
    last = torch.randint(0, 30000, (max_num_reqs,), generator=g).to(dev).int()
    nextp = torch.randint(0, 30000, (max_num_reqs,), generator=g).to(dev).int()
    # num_rejected < num_ctx always: a step verifies num_ctx positions and
    # accepts at least one, so rejections can never exceed the context span.
    nrej = torch.randint(0, max(1, qlen), (num_reqs,), generator=g)
    nsamp = (nq - nrej).clamp_(min=1).to(dev).int()
    nrej = nrej.to(dev).int()
    bt = torch.randint(0, 5000, (max_num_reqs, bt_stride), generator=g).to(dev).int()
    bt[bt < 200] = 0  # exercise the null-block guard
    args = [outs["input_ids"], outs["qpos"], outs["qsl"], outs["seq_lens"], outs["qslot"],
            outs["cpos"], outs["cslot"], outs["sidx"], outs["spos"], outs["sidxmap"],
            tpos, tqsl, idxmap, last, nextp, nsamp, nrej, bt, bt.stride(0),
            151329, block_size, nq, nsteps, max_num_reqs, max_num_tokens, max_model_len]
    BLOCK = min(256, triton.next_power_of_2(qlen + nq))
    nblocks = triton.cdiv(qlen + nq, BLOCK)
    if is_new:
        kernel[(num_reqs, nblocks)](*args, 0, SAMPLE_FROM_ANCHOR=False,
                                    PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=1,
                                    CP_INTERLEAVE=1, BLOCK_SIZE=BLOCK)
    else:
        kernel[(num_reqs, nblocks)](*args, SAMPLE_FROM_ANCHOR=False,
                                    GUARD_NULL_BLOCK=True,
                                    PAD_SLOT_ID=PAD_SLOT_ID, BLOCK_SIZE=BLOCK)
    return outs

cfgs = []
for block_size in (16, 64, 128):
    for qlen in (1, 4, 8, 13):
        for seed in range(6):
            cfgs.append(dict(num_reqs=8, qlen=qlen, nq=8, nsteps=7, block_size=block_size,
                             max_num_reqs=16, max_num_tokens=16 * 8, max_model_len=400000,
                             bt_stride=8192, seed=seed))
bad = 0
for c in cfgs:
    a = run(new_k, True, **c)
    b = run(old_k, False, **c)
    for k in a:
        if not torch.equal(a[k], b[k]):
            bad += 1
            print(f"MISMATCH {k} block_size={c['block_size']} qlen={c['qlen']} seed={c['seed']}")
            break
print(f"\nconfigs: {len(cfgs)}   mismatched: {bad}")
print("CP1 GATE:", "PASS - bit-identical" if bad == 0 else "FAIL")
sys.exit(1 if bad else 0)
