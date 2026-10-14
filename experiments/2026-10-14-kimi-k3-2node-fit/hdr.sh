#!/usr/bin/env bash
f="$1"; out="$2/$f.json"; url="https://huggingface.co/moonshotai/Kimi-K3/resolve/main/$f"
n=$(curl -sL -m 60 -r 0-7 "$url" | od -An -t u8 | tr -d ' ')
curl -sL -m 120 -r 8-$((n+7)) "$url" -o "$out"
