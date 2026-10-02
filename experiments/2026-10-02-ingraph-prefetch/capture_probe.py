"""Debug probe: log the first point where a stream's capture is invalidated.

Imported by a temporary hook in tiered_moe_execution.py (never committed).
"""
import ctypes
import logging
import os
import traceback

import torch

_lib = ctypes.CDLL("libcudart.so.13")
_seen = set()
log = logging.getLogger("capture_probe")
NAMES = {0: "none", 1: "active", 2: "INVALIDATED"}


def status(stream: torch.cuda.Stream) -> str:
    out = ctypes.c_int(-1)
    err = _lib.cudaStreamIsCapturing(ctypes.c_void_p(stream.cuda_stream), ctypes.byref(out))
    return NAMES.get(out.value, str(out.value)) if err == 0 else f"err{err}"


def check(where: str, layer_id, *streams) -> None:
    states = [status(s) for s in streams if s is not None]
    key = (where, layer_id)
    bad = any(s == "INVALIDATED" or s.startswith("err") for s in states)
    if os.environ.get("CAPTURE_PROBE_ALL") or (bad and not _seen):
        _seen.add(key)
        print(f"[capture_probe rank? dev={torch.cuda.current_device()}] "
              f"{where} layer={layer_id} states={states}", flush=True)
        if bad:
            print("".join(traceback.format_stack(limit=12)), flush=True)
