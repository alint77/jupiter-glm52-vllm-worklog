#!/usr/bin/env python3
"""Generate the narration, one clip per segment, with ElevenLabs.

Reads narration_segments.json, writes voice/seg/<Scene>_<i>.mp3 and
voice/segments.json ({scene: [{"text", "file", "dur"}]}) for the scenes to
time their beats against. A clip is regenerated only when its text changes.
The API key is read from ~/.config/elevenlabs/key.
"""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Brian (ElevenLabs stock voice)
MODEL = "eleven_v4"
KEY = (Path.home() / ".config/elevenlabs/key").read_text().strip()
OUT = HERE / "voice" / "seg"
OUT.mkdir(parents=True, exist_ok=True)


def tts(text, prev, nxt, path):
    body = {"text": text, "model_id": MODEL, "previous_text": prev, "next_text": nxt}
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
        data=json.dumps(body).encode(),
        headers={"xi-api-key": KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        path.write_bytes(r.read())


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout)


segs = json.loads((HERE / "narration_segments.json").read_text())
flat = [(s, i, t) for s, ts in segs.items() for i, t in enumerate(ts)]
result, spent = {}, 0
for k, (scene, i, text) in enumerate(flat):
    tag = hashlib.sha1(f"{VOICE_ID}|{MODEL}|{text}".encode()).hexdigest()[:10]
    path = OUT / f"{scene}_{i:02d}_{tag}.mp3"
    if not path.exists():
        prev = flat[k - 1][2] if k else ""
        nxt = flat[k + 1][2] if k + 1 < len(flat) else ""
        tts(text, prev, nxt, path)
        spent += len(text)
        print(f"generated {path.name} ({len(text)} chars)", flush=True)
    result.setdefault(scene, []).append({"text": text, "file": str(path.relative_to(HERE)), "dur": round(duration(path), 3)})
(HERE / "voice" / "segments.json").write_text(json.dumps(result, indent=1))
print(f"{spent} chars spent; total narration {sum(d['dur'] for v in result.values() for d in v):.1f} s")
