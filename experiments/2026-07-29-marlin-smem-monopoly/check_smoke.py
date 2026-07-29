"""Fail closed if the greedy eight-token continuation is not the expected one."""

import json
import sys

path, tag = sys.argv[1], sys.argv[2]
expected = " Paris. Distance from Paris to Lyon is"
text = json.load(open(path))["choices"][0]["text"]
print(f"{tag} smoke: {text!r}")
sys.exit(0 if text == expected else 1)
