"""Run the SAME inputs through our vLLM DFlash2 modules (CPU, real weights)
and diff against the z-lab reference saved by probe_offline.py.

Instantiates the real module classes directly (no engine) so the comparison
covers our conv, attention math, norms and residual flow exactly as written.
"""
import json, torch, torch.nn.functional as F
from safetensors import safe_open
import sys
sys.path.insert(0, '/e/project1/profound/alint77/vllm')

CKPT = "/e/project1/profound/alint77/models/GLM-5.3-DFlash2"
cfg = json.load(open(f"{CKPT}/config.json"))
dfl = cfg["dflash_config"]
H, HD = cfg["hidden_size"], cfg["head_dim"]
NH, NKV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
EPS, L = cfg["rms_norm_eps"], cfg["num_hidden_layers"]
GS, TAPS, BLOCK = dfl["conv_group_size"], dfl["conv_kernel_size"], dfl["block_size"]
THETA = cfg["rope_parameters"]["rope_theta"]

W = {}
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()

d = torch.load("/tmp/dflash2_ref.pt")
ref_hidden, emb = d["ref_hidden"], d["emb"]
target_hidden, ctx_pos, q_pos = d["target_hidden"], d["ctx_pos"], d["q_pos"]
NCTX, NQ = target_hidden.shape[0], emb.shape[0]

# --- our production conv, imported directly ---
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv

def our_conv(x, delta, base):
    groups = H // GS
    return _grouped_conv(x, delta, base, BLOCK, groups, GS, TAPS)

def rms(x, w, eps=EPS):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w

inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
fr = torch.einsum("i,j->ij", torch.arange(4096).float(), inv)
COS_SIN = torch.cat((fr.cos(), fr.sin()), dim=-1)

# --- our production rope, imported directly ---
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
def our_rope(x, pos):
    cs = COS_SIN.index_select(0, pos.long())
    cos, sin = cs.chunk(2, dim=-1)
    return ApplyRotaryEmb.forward_static(x, cos, sin, True)

# Replicate OUR serving structure: context K/V precomputed and cached (with
# our fused path's ordering), query rows attend over [context, query].
ctx = rms(target_hidden @ W["fc.weight"].T, W["hidden_norm.weight"])

h = emb.clone()
residual = None
for li in range(L):
    p = f"layers.{li}."
    # our layer: fused_add_rms_norm keeps residual separate
    if residual is None:
        residual = h
        h = rms(h, W[p+"input_layernorm.weight"])
    else:
        h = h + residual
        residual = h
        h = rms(h, W[p+"input_layernorm.weight"])
    groups = H // GS
    dyn = (h @ W[p+"attention_conv.kernel_projection.weight"].T).reshape(NQ, 2, TAPS, groups)
    h = our_conv(h, dyn[:, 0], W[p+"attention_conv.base_kernel"][0])
    ak = dyn[:, 1]
    q = (h @ W[p+"self_attn.q_proj.weight"].T).view(NQ, NH, HD)
    q = rms(q, W[p+"self_attn.q_norm.weight"])
    k_q = (h @ W[p+"self_attn.k_proj.weight"].T).view(NQ, NKV, HD)
    k_q = rms(k_q, W[p+"self_attn.k_norm.weight"])
    v_q = (h @ W[p+"self_attn.v_proj.weight"].T).view(NQ, NKV, HD)
    # context K/V through OUR fused precompute ordering: kv_proj then k_norm then rope
    k_ctx = (ctx @ W[p+"self_attn.k_proj.weight"].T).view(NCTX, NKV, HD)
    k_ctx = rms(k_ctx, W[p+"self_attn.k_norm.weight"])
    v_ctx = (ctx @ W[p+"self_attn.v_proj.weight"].T).view(NCTX, NKV, HD)
    q = our_rope(q, q_pos)
    k_q = our_rope(k_q, q_pos)
    k_ctx = our_rope(k_ctx, ctx_pos)
    k = torch.cat([k_ctx, k_q], 0); v = torch.cat([v_ctx, v_q], 0)
    g = NH // NKV
    q4 = q.view(1, NQ, NKV, g, HD).permute(0,2,3,1,4).reshape(1, NH, NQ, HD)
    k4 = k.transpose(0,1).repeat_interleave(g,0).unsqueeze(0)
    v4 = v.transpose(0,1).repeat_interleave(g,0).unsqueeze(0)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.view(1, NKV, g, NQ, HD).permute(0,3,1,2,4).reshape(NQ, NH*HD)
    o = o @ W[p+"self_attn.o_proj.weight"].T
    o = our_conv(o, ak, W[p+"attention_conv.base_kernel"][1])
    h = o
    h = h + residual
    residual = h
    h = rms(h, W[p+"post_attention_layernorm.weight"])
    dyn = (h @ W[p+"mlp_conv.kernel_projection.weight"].T).reshape(NQ, 2, TAPS, groups)
    h = our_conv(h, dyn[:, 0], W[p+"mlp_conv.base_kernel"][0])
    mk = dyn[:, 1]
    gate = h @ W[p+"mlp.gate_proj.weight"].T
    up = h @ W[p+"mlp.up_proj.weight"].T
    m = (F.silu(gate) * up) @ W[p+"mlp.down_proj.weight"].T
    m = our_conv(m, mk, W[p+"mlp_conv.base_kernel"][1])
    h = m
h = h + residual
our_hidden = rms(h, W["norm.weight"])

diff = (our_hidden - ref_hidden).abs()
rel = diff.max().item() / ref_hidden.abs().max().item()
print(f"final hidden  maxdiff {diff.max().item():.4e}  rel {rel:.4e}")
print(f"  ref norm {ref_hidden.norm().item():.3f}  ours {our_hidden.norm().item():.3f}")
print("  per-row maxdiff:", [f"{v:.3e}" for v in diff.max(dim=-1).values.tolist()])
