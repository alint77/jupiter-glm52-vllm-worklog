"""Run tests/evals/gsm8k/gsm8k_eval.py against a server that requires a key.

The eval opens a bare aiohttp.ClientSession; this installs VLLM_API_KEY as a
default Authorization header before handing over to its CLI.
"""

import os
import sys

import aiohttp

sys.path.insert(0, "tests/evals/gsm8k")
import gsm8k_eval  # noqa: E402

_session = aiohttp.ClientSession


class _AuthSession(_session):
    def __init__(self, *args, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
        super().__init__(*args, headers=headers, **kwargs)


aiohttp.ClientSession = _AuthSession
gsm8k_eval.aiohttp.ClientSession = _AuthSession
gsm8k_eval.main()
