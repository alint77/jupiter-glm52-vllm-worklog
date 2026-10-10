#!/usr/bin/env bash
# cost of the shared-memory proxy fence (race fix) at M=32 or M=64: e76.sh 32|64
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
V="td_v56:TD_MAX_TOKENS=64"
if [[ $1 == 32 ]]; then cells="96,8 108,16 120,32"; else cells="132,16 144,24 156,48 156,60"; fi
./ab.sh e76-$1 $1 "$cells" "$V|--shared 1" "$V TD_SFENCE_PROD|--shared 1" "$V TD_SFENCE_CONS|--shared 1" "$V TD_SFENCE|--shared 1"
echo E76_DONE
