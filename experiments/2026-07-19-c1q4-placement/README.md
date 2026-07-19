# MTP3 c1/q4 domain-aware expert placement

This experiment replaces the original synthetic, pre-MTP routing profile with
target-verification traces from MTP3 at concurrency one (q4). The workload is
balanced across Python, PyTorch, machine learning, and mathematics: four
training and two held-out prompts per domain, each with 256 forced output
tokens.

The comparison holds the 2,870-HBM-expert/rank budget fixed:

1. Existing synthetic/pre-MTP profile.
2. Domain-mixed MTP3 per-expert profile.
3. Layer-concentrated hybrid profiles. These reserve complete layers in HBM
   when eliminating the second tier is worth a configurable mixed-layer
   penalty, then spend the remaining slots on the most frequent experts.

Route capture is decode-only and retains all four target routes in every MTP
verification step, including rejected drafts. Normal routed-expert responses
remain unchanged unless the internal `return_rejected_routed_experts` request
flag is set.

Status: implementation and capture qualification in progress.
