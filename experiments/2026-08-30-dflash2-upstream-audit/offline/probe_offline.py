"""Offline DFlash2 draft check: build the real draft model from the real
checkpoint, run a pure-torch reference forward (z-lab semantics) and the
vLLM module stack on identical inputs, and diff every stage.

Runs on CPU with real weights: no server, no GPU, no scheduler. Any layout /
reshape / slice bug in the draft's own math shows up here.
"""
import json, torch, torch.nn.functional as F
from safetensors import safe_open

CKPT = "/e/project1/profound/alint77/models/GLM-5.3-DFlash2"
cfg = json.load(open(f"{CKPT}/config.json"))
dfl = cfg["dflash_config"]
H = cfg["hidden_size"]; HD = cfg["head_dim"]
NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
EPS = cfg["rms_norm_eps"]; L = cfg["num_hidden_layers"]
GS = dfl["conv_group_size"]; TAPS = dfl["conv_kernel_size"]
BLOCK = dfl["block_size"]; RANK = dfl["selector_rank"]; K = dfl["selector_top_k"]
THETA = cfg["rope_parameters"]["rope_theta"]
print(f"cfg: H={H} HD={HD} NH={NH} NKV={NKV} L={L} block={BLOCK} K={K} rank={RANK}")

W = {}
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()   # fp32 for a clean reference

torch.manual_seed(0)
NCTX, NQ = 24, BLOCK          # context rows, query rows (1 bonus + 7 masks)
target_hidden = torch.randn(NCTX, H * len(dfl["target_layer_ids"])) * 0.05
emb = torch.randn(NQ, H) * 0.05
ctx_pos = torch.arange(NCTX)
q_pos = torch.arange(NCTX, NCTX + NQ)

def rms(x, w, eps=EPS):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w

inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
tt = torch.arange(4096).float()
fr = torch.einsum("i,j->ij", tt, inv)
COS_SIN = torch.cat((fr.cos(), fr.sin()), dim=-1)

def rope(x, pos):           # x: [N, heads, HD] neox
    cs = COS_SIN.index_select(0, pos.long())
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos[:, None, :]; sin = sin[:, None, :]
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1*cos - x2*sin, x2*cos + x1*sin), dim=-1)

def conv(x, delta, base):   # z-lab semantics, one block
    n, hidden = x.shape
    groups = hidden // GS
    blocks = x.view(n, groups, GS)
    dyn = delta.view(n, TAPS, groups, 1)
    out = torch.zeros_like(blocks)
    for off in range(TAPS):
        vals = blocks if off == 0 else F.pad(blocks[:-off], (0,0,0,0,off,0))
        out = out + base[off].view(1,1,groups,GS) * vals
        out = torch.addcmul(out, dyn[:, off], vals)
    return out.view_as(x)

# ---- reference forward (z-lab dflash/model.py semantics) ----
ctx = rms(target_hidden @ W["fc.weight"].T, W["hidden_norm.weight"])
h = emb.clone()
for li in range(L):
    p = f"layers.{li}."
    residual = h
    h = rms(h, W[p+"input_layernorm.weight"])
    groups = H // GS
    dyn = (h @ W[p+"attention_conv.kernel_projection.weight"].T).view(NQ, 2, TAPS, groups)
    h = conv(h, dyn[:, 0], W[p+"attention_conv.base_kernel"][0])
    ak = dyn[:, 1]
    q = (h @ W[p+"self_attn.q_proj.weight"].T).view(NQ, NH, HD)
    q = rms(q, W[p+"self_attn.q_norm.weight"])
    k_ctx = (ctx @ W[p+"self_attn.k_proj.weight"].T).view(NCTX, NKV, HD)
    k_q   = (h   @ W[p+"self_attn.k_proj.weight"].T).view(NQ,  NKV, HD)
    v_ctx = (ctx @ W[p+"self_attn.v_proj.weight"].T).view(NCTX, NKV, HD)
    v_q   = (h   @ W[p+"self_attn.v_proj.weight"].T).view(NQ,  NKV, HD)
    k = rms(torch.cat([k_ctx, k_q], 0), W[p+"self_attn.k_norm.weight"])
    v = torch.cat([v_ctx, v_q], 0)
    q = rope(q, q_pos)
    k = rope(k, torch.cat([ctx_pos, q_pos]))
    g = NH // NKV
    q4 = q.view(1, NQ, NKV, g, HD).permute(0,2,3,1,4).reshape(1, NH, NQ, HD)
    k4 = k.transpose(0,1).repeat_interleave(g, 0).unsqueeze(0)
    v4 = v.transpose(0,1).repeat_interleave(g, 0).unsqueeze(0)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.view(1, NKV, g, NQ, HD).permute(0,3,1,2,4).reshape(NQ, NH*HD)
    o = o @ W[p+"self_attn.o_proj.weight"].T
    o = conv(o, ak, W[p+"attention_conv.base_kernel"][1])
    h = residual + o
    residual = h
    h = rms(h, W[p+"post_attention_layernorm.weight"])
    dyn = (h @ W[p+"mlp_conv.kernel_projection.weight"].T).view(NQ, 2, TAPS, groups)
    h = conv(h, dyn[:, 0], W[p+"mlp_conv.base_kernel"][0])
    mk = dyn[:, 1]
    gate = h @ W[p+"mlp.gate_proj.weight"].T
    up   = h @ W[p+"mlp.up_proj.weight"].T
    m = (F.silu(gate) * up) @ W[p+"mlp.down_proj.weight"].T
    m = conv(m, mk, W[p+"mlp_conv.base_kernel"][1])
    h = residual + m
ref_hidden = rms(h, W["norm.weight"])
print("reference final hidden:", tuple(ref_hidden.shape),
      "norm", ref_hidden.norm().item())
torch.save({"ref_hidden": ref_hidden, "emb": emb, "target_hidden": target_hidden,
            "ctx_pos": ctx_pos, "q_pos": q_pos},
           "/tmp/dflash2_ref.pt")
print("saved /tmp/dflash2_ref.pt")
