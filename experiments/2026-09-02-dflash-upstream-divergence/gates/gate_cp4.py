"""CP>1 gate: the ported kernel's ctx/query slots must match an independent
torch reference of the DCP round-robin mapping (block_table.py:283-300)."""
import sys, os
sys.path.insert(0, os.environ["CLAUDE_JOB_DIR"] + "/tmp")
import torch, triton
from knew import _prepare_dflash_inputs_kernel as K

PAD = -1
dev = "cuda"

def ref_slot(pos, block_id, block_size, cp_rank, cp_size, cp_int):
    offs = pos % (block_size * cp_size)
    if cp_size == 1:
        return block_id * block_size + offs
    is_local = (offs // cp_int) % cp_size == cp_rank
    rounds = offs // (cp_int * cp_size)
    rem = offs % cp_int
    local = rounds * cp_int + rem
    return torch.where(is_local, block_id * block_size + local,
                       torch.full_like(pos, PAD))

def check(cp_size, cp_rank, cp_int, block_size, qlen, nq, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_reqs, nsteps, mnr, bts = 8, 7, 16, 8192
    mnt = mnr * nq
    total = num_reqs * qlen
    o = dict(
        ii=torch.zeros(mnt, dtype=torch.int32, device=dev),
        qpos=torch.zeros(mnt, dtype=torch.int64, device=dev),
        qsl=torch.zeros(mnr + 1, dtype=torch.int32, device=dev),
        sl=torch.zeros(mnr, dtype=torch.int32, device=dev),
        qslot=torch.zeros(mnt, dtype=torch.int64, device=dev),
        cpos=torch.zeros(total + 64, dtype=torch.int64, device=dev),
        cslot=torch.zeros(total + 64, dtype=torch.int64, device=dev),
        si=torch.zeros(mnr * nsteps, dtype=torch.int64, device=dev),
        sp=torch.zeros(mnr * nsteps, dtype=torch.int64, device=dev),
        sim=torch.zeros(mnr * nsteps, dtype=torch.int32, device=dev),
    )
    base = torch.randint(1000, 5000, (num_reqs,), generator=g)
    tpos = torch.cat([base[i] + torch.arange(qlen) for i in range(num_reqs)]).to(dev).long()
    tqsl = (torch.arange(num_reqs + 1) * qlen).to(dev).int()
    idxmap = torch.arange(num_reqs, device=dev, dtype=torch.int32)
    last = torch.randint(0, 30000, (mnr,), generator=g).to(dev).int()
    nextp = torch.randint(0, 30000, (mnr,), generator=g).to(dev).int()
    nrej = torch.randint(0, max(1, qlen), (num_reqs,), generator=g)
    nsamp = (nq - nrej).clamp_(min=1).to(dev).int()
    nrej = nrej.to(dev).int()
    bt = torch.randint(0, 5000, (mnr, bts), generator=g).to(dev).int()
    bt[bt < 200] = 0
    args = [o["ii"], o["qpos"], o["qsl"], o["sl"], o["qslot"], o["cpos"], o["cslot"],
            o["si"], o["sp"], o["sim"], tpos, tqsl, idxmap, last, nextp, nsamp, nrej,
            bt, bt.stride(0), 151329, block_size, nq, nsteps, mnr, mnt, 400000]
    BLOCK = min(256, triton.next_power_of_2(qlen + nq))
    K[(num_reqs, triton.cdiv(qlen + nq, BLOCK))](
        *args, cp_rank, SAMPLE_FROM_ANCHOR=False, PAD_SLOT_ID=PAD,
        CP_SIZE=cp_size, CP_INTERLEAVE=cp_int, BLOCK_SIZE=BLOCK)

    # --- reference for the valid context rows ---
    errs = []
    for r in range(num_reqs):
        nrj = int(nrej[r]); nvalid = qlen - nrj
        for j in range(nvalid):
            p = tpos[r * qlen + j]
            bn = min(int(p) // (block_size * cp_size), bts - 1)
            bid = bt[r, bn].long()
            exp = PAD if bid == 0 else int(ref_slot(p, bid, block_size, cp_rank, cp_size, cp_int))
            got = int(o["cslot"][r * qlen + j])
            if got != exp:
                errs.append(("ctx", r, j, got, exp))
        # query rows
        lvp = int(tpos[r * qlen + nvalid - 1])
        for off in range(nq):
            qp = lvp + 1 + off
            bn = min(qp // (block_size * cp_size), bts - 1)
            bid = bt[r, bn].long()
            exp = PAD if bid == 0 else int(ref_slot(torch.tensor(qp, device=dev), bid,
                                                    block_size, cp_rank, cp_size, cp_int))
            got = int(o["qslot"][r * nq + off])
            if got != exp:
                errs.append(("q", r, off, got, exp))
    return errs

total_err = 0; n = 0
for cp_size, cp_int in ((2, 1), (4, 1), (4, 8), (8, 4)):
    for cp_rank in range(cp_size):
        for block_size in (16, 64):
            for qlen in (4, 8):
                n += 1
                e = check(cp_size, cp_rank, cp_int, block_size, qlen, 8, seed=cp_rank + qlen)
                if e:
                    total_err += len(e)
                    print(f"cp_size={cp_size} rank={cp_rank} int={cp_int} bs={block_size} "
                          f"qlen={qlen}: {len(e)} mismatches, first={e[0]}")
print(f"\nconfigs: {n}   total mismatches: {total_err}")
# PAD coverage sanity: with cp_size>1 a rank owns only a fraction of positions.
print("CP>1 GATE:", "PASS" if total_err == 0 else "FAIL")
sys.exit(1 if total_err else 0)
