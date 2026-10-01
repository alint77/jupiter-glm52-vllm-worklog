#!/usr/bin/env python3
"""Lay the narration clips onto the rendered video at the logged cue times.

    mix.py <quality: l|m|h|k>
Uses voice/segments.json (clip durations), voice/cues/<Scene>.json (when each
beat started, scene-relative, written during the render) and the per-scene
videos from render.sh; writes tiered_moe_<q>_narrated.mp4 in the media dir.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MEDIA = Path(f"/e/fscratch/profound/{os.environ['USER']}/manim-media")
SUB = {"l": "480p15", "m": "720p30", "h": "1080p60", "k": "2160p60"}
SCENES = ["S1Hardware", "S2Problem", "S3StockVllm", "S4Layout", "S4bPrefetch", "S5Overlap",
          "S6Problem", "S7Frequency", "S8Imbalance", "S9Replicas", "S10Result"]


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout)


q = sys.argv[1] if len(sys.argv) > 1 else "h"
segs = json.loads((HERE / "voice/segments.json").read_text())
inputs, filters, labels, offset = [], [], [], 0.0
for scene in SCENES:
    video = MEDIA / "scenes" / scene / "videos/tiered_moe" / SUB[q] / f"{scene}.mp4"
    cues = json.loads((HERE / "voice/cues" / f"{scene}.json").read_text())
    assert len(cues) == len(segs[scene]), f"{scene}: {len(cues)} cues, {len(segs[scene])} clips"
    for t, seg in zip(cues, segs[scene]):
        k = len(inputs) // 2
        inputs += ["-i", str(HERE / seg["file"])]
        ms = int(round((offset + t) * 1000))
        filters.append(f"[{k}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
        labels.append(f"[a{k}]")
    offset += duration(video)
filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
               f"apad,atrim=0:{offset:.3f}[mix]")
audio = MEDIA / f"narration_{q}.m4a"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
                "-map", "[mix]", "-c:a", "aac", "-b:a", "192k", str(audio)], check=True)
out = MEDIA / f"tiered_moe_{q}_narrated.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(MEDIA / f"tiered_moe_{q}.mp4"), "-i", str(audio),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest", str(out)], check=True)
print(out, f"{offset:.1f} s")
