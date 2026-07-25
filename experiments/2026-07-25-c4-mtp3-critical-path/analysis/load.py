import gzip, json, os

DIRS = {
    1: "/e/scratch/profound/naeimitabiei1/glm52-c4-mtp1-edge-profile-1022400",
    2: "/e/scratch/profound/naeimitabiei1/glm52-c4-mtp2-edge-profile-1022401",
    3: "/e/scratch/profound/naeimitabiei1/glm52-c4-mtp3-edge-profile-1022402",
}
TRACE_DIR = DIRS[int(os.environ.get("MTP", "3"))]

def load(rank, depth=None):
    d = DIRS[depth] if depth else TRACE_DIR
    fn = [f for f in os.listdir(d) if f"rank{rank}." in f and f.endswith(".gz")]
    assert len(fn) == 1, (d, fn)
    with gzip.open(os.path.join(d, fn[0]), "rt") as f:
        return json.load(f)
