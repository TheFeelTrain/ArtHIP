from vsscale import ArtCNN, Backend
from vstools import core, vs
import os
import sys

core.max_cache_size = 1024*48

# Backend selection:  VS_BACKEND=hip|migx  (defaults to hip).
# vspipe R79 does not pass script args to Python (sys.argv stays empty), so
# an env var is the reliable mechanism; sys.argv is kept as a fallback for
# vspipe versions that do forward positional args.
backend = os.environ.get("VS_BACKEND") or (sys.argv[1] if len(sys.argv) > 1 else "hip")
backend = backend.lower()
if backend in ("migx", "migraphx"):
    b = Backend.MIGX(fp16=True, num_streams=2)
elif backend in ("hip",):
    b = Backend.HIP(fp16=True, num_streams=2)
else:
    sys.exit(f"unknown backend: {backend} (use 'hip' or 'migx')")

clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=500)
ArtCNN.R8F64(backend=b).scale(clip, clip.width * 2, clip.height * 2).set_output()