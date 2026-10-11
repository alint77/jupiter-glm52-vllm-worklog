import re, sys
b = {}
for l in open(sys.argv[1]):
    m = re.match(r'(vllm:\w+)_bucket\{.*le="([^"]+)".*\} ([\d.e+]+)', l)
    if m: b.setdefault(m[1], []).append((float(m[2]), float(m[3])))
for name in ["time_to_first_token_seconds", "request_queue_time_seconds", "request_prefill_time_seconds", "inter_token_latency_seconds", "request_time_per_output_token_seconds", "e2e_request_latency_seconds", "request_prompt_tokens", "request_prefill_kv_computed_tokens", "request_generation_tokens"]:
    v = sorted(b.get("vllm:" + name, [])); n = v[-1][1]
    out = []
    for q in (.5, .9, .99):
        prev = (0, 0)
        for le, c in v:
            if c >= q * n: out.append(f"p{int(q*100)}<={le:g}"); break
    print(f"{name:42s} n={n:.0f} " + " ".join(out))
