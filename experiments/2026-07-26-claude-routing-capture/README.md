# Claude Code routing capture

This experiment records routed-expert IDs from natural Claude Code turns on
GLM-5.2 AutoRound W4G64 MTP3. It does not store prompts, responses, tool
arguments, or repository contents. Each trace starts at generation, avoiding
repeated counting of the conversation prefix.

The capture host uses c1/TP4/EP4 because routed-expert return is not supported
with DCP4. Claude chooses the natural stopping point; no 256-token output cap
is applied. MTP verification routes include rejected drafts.

Start a four-hour capture host and use it normally:

```bash
./claude-local-capture.sh
```

After collecting several representative sessions:

```bash
./claude-local-capture.sh --grid
```

The grid, ranking, heatmap, and summary are written under the capture state
directory for the active job.

Implementation source: vLLM fork commit `af5587af3`. Focused route-writer
pytest passed, all commit hooks passed, and the grid builder completed a
synthetic end-to-end check with the expected `4 * 75 * 8 = 2,400` routes.

Job `1047751` is queued behind four-node NanoPLM job `1047419`, keeping active
Booster allocation within the eight-node limit. Live capture validation will
begin after that dependency completes.
