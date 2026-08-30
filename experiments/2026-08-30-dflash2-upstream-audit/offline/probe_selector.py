"""Selector stage: z-lab reference select() vs our production _score_edges +
walk, on real checkpoint weights."""
import json, torch, sys
sys.path.insert(0,'/e/project1/profound/alint77/vllm')
from safetensors import safe_open

CKPT="/e/project1/profound/alint77/models/GLM-5.3-DFlash2"
cfg=json.load(open(f"{CKPT}/config.json")); dfl=cfg["dflash_config"]
H=cfg["hidden_size"]; K=dfl["selector_top_k"]; RANK=dfl["selector_rank"]
V=cfg["vocab_size"]; STEPS=dfl["block_size"]-1
W={}
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    for k in ("candidate_selector.predecessor_codebook",
              "candidate_selector.successor_codebook",
              "candidate_selector.hidden_projection.weight"):
        W[k]=f.get_tensor(k).float()
P=W["candidate_selector.predecessor_codebook"]
S=W["candidate_selector.successor_codebook"]
HP=W["candidate_selector.hidden_projection.weight"]

torch.manual_seed(0)
B=3
hidden=torch.randn(B,STEPS,H)*0.05
logits=torch.randn(B,STEPS,V)*2.0
anchor=torch.randint(0,V,(B,))

# ---- reference select() (z-lab) ----
unary_r, cand_r = torch.topk(logits, K, dim=-1, sorted=False)
hp = hidden @ HP.T
pred = anchor
ref_path=[]
ref_scores=[]
for pos in range(STEPS):
    sc = unary_r[:,pos] + torch.einsum("br,bkr->bk",
        P[pred]*hp[:,pos], S[cand_r[:,pos]])
    ref_scores.append(sc)
    idx = torch.argmax(sc, dim=-1)
    pred = cand_r[:,pos].gather(-1, idx[:,None])[:,0]
    ref_path.append(pred)
ref_path=torch.stack(ref_path,1)

# ---- ours: _score_edges (production) then greedy walk ----
from vllm.model_executor.models.qwen3_dflash2 import _score_edges
# our candidates come from a SORTED topk (flashinfer sorted=True / torch.topk)
unary_o, cand_o = torch.topk(logits, K, dim=-1)   # sorted=True default
scores_o = _score_edges(P, S, cand_o, unary_o, hp, anchor, K)
prev=torch.zeros(B,dtype=torch.long)
our_path=[]
for step in range(STEPS):
    rows=torch.arange(B)
    s=scores_o[rows,step,prev,:]
    idx=s.argmax(-1)
    our_path.append(cand_o[rows,step,:].gather(-1,idx[:,None])[:,0])
    prev=idx
our_path=torch.stack(our_path,1)

print("candidate sets identical (as sets):",
      all(set(cand_r[b,p].tolist())==set(cand_o[b,p].tolist())
          for b in range(B) for p in range(STEPS)))
print("reference path :", ref_path.tolist())
print("our path       :", our_path.tolist())
print("PATHS MATCH    :", torch.equal(ref_path, our_path))
# score agreement on the shared step-0 row (prev=0 vs reference anchor row)
d0=(scores_o[:,0,0,:] - ref_scores[0]).abs()
print("step0 score maxdiff (sorted vs unsorted candidate order may differ):",
      d0.max().item())
