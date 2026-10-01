#!/usr/bin/env bash
# Render every scene and join them:  bash render.sh [l|m|h|k]  (default h)
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-h}"
S=/e/software/default/stages/2026/software
export PKG_CONFIG_PATH="$(ls -d $S/*/*GCCcore-14.3.0/lib*/pkgconfig 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="$(ls -d $S/*/*GCCcore-14.3.0/lib64 $S/*/*GCCcore-14.3.0/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export PATH="$S/FFmpeg/7.1.2-GCCcore-14.3.0/bin:$PATH"
MANIM=/e/fscratch/profound/${USER}/venvs/manim-venv/bin/manim
MEDIA=/e/fscratch/profound/${USER}/manim-media
SCENES=(S1Hardware S2Problem S3StockVllm S4Layout S5Overlap S6Problem S7Frequency S8Imbalance S9Replicas S10Result)
for s in "${SCENES[@]}"; do
  "$MANIM" -q"$Q" --media_dir "$MEDIA" --disable_caching -v WARNING tiered_moe.py "$s"
done
dir=$(dirname "$(find "$MEDIA/videos/tiered_moe" -name S1Hardware.mp4 -newer tiered_moe.py | head -1)")
: > "$MEDIA/list.txt"
for s in "${SCENES[@]}"; do echo "file '$dir/$s.mp4'" >> "$MEDIA/list.txt"; done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$MEDIA/list.txt" -c copy "$MEDIA/tiered_moe_$Q.mp4"
echo "$MEDIA/tiered_moe_$Q.mp4"
