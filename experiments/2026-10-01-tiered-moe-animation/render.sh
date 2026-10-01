#!/usr/bin/env bash
# Render scenes in parallel and join all of them:
#   bash render.sh [l|m|h|k] [Scene ...]
# With scene names, only those are re-rendered; the others are reused from the
# last render at that quality. JOBS sets the parallelism (default: all at once).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-h}"; shift || true
S=/e/software/default/stages/2026/software
export PKG_CONFIG_PATH="$(ls -d $S/*/*GCCcore-14.3.0/lib*/pkgconfig 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="$(ls -d $S/*/*GCCcore-14.3.0/lib64 $S/*/*GCCcore-14.3.0/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export PATH="$S/FFmpeg/7.1.2-GCCcore-14.3.0/bin:$PATH"
MANIM=/e/fscratch/profound/${USER}/venvs/manim-venv/bin/manim
MEDIA=/e/fscratch/profound/${USER}/manim-media
ALL=(S0Intro S1Hardware S2Problem S3StockVllm S4Layout S4bPrefetch S5Overlap S6Problem S7Frequency S8Imbalance S9Replicas S10Result S11Outro)
TODO=("${@:-${ALL[@]}}")
# One media dir per scene: parallel runs would race on the shared text cache.
printf '%s\n' "${TODO[@]}" | xargs -P "${JOBS:-${#TODO[@]}}" -I{} \
  "$MANIM" -q"$Q" --media_dir "$MEDIA/scenes/{}" --disable_caching -v ERROR --progress_bar none tiered_moe.py {}
case "$Q" in l) sub=480p15 ;; m) sub=720p30 ;; h) sub=1080p60 ;; k) sub=2160p60 ;; esac
: > "$MEDIA/list-$Q.txt"
for s in "${ALL[@]}"; do echo "file '$MEDIA/scenes/$s/videos/tiered_moe/$sub/$s.mp4'" >> "$MEDIA/list-$Q.txt"; done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$MEDIA/list-$Q.txt" -c copy "$MEDIA/tiered_moe_$Q.mp4"
echo "$MEDIA/tiered_moe_$Q.mp4"
