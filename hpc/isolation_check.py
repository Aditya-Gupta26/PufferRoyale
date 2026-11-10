"""Isolation check for every PufferRoyale job on the cluster (docs/HPC_HANDOFF.md §4d, rule R10).

Run INSIDE the container with our checkout as the working directory (the `run` helper of
hpc/lib.sh does both):    run python hpc/isolation_check.py

It fails (exit 1) unless:
  * no interpreter / package path lies inside $PD_REF (the read-only PufferDrive reference);
  * numpy is 1.x;
  * pufferlib/pufferl.py is upstream PufferLib 3.0 @ 3b5c604 (sha256 below);
  * the `pufferroyale` imported is the one in the working directory (our checkout), so an installed
    copy (e.g. the image's /opt/pufferroyale) can never be trained by mistake.
It also prints whether $PD_REF is visible inside the container at all (it should not be: our jobs
bind only $PR_ROOT) and the versions to record in docs/EXPERIMENTS.md.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())          # as `python -` would: the checkout first

import numpy  # noqa: E402
import pufferlib  # noqa: E402
import pufferlib.pufferl  # noqa: E402
import torch  # noqa: E402

import pufferroyale  # noqa: E402

UPSTREAM_PUFFERL_SHA256 = "ebe89e96a2515e7b0370d7b4833ae64bb46c1a68c20d13fd014a10abc4e33323"

ref = os.path.realpath(os.environ.get("PD_REF", "/nonexistent"))
paths = {"python": sys.executable, "pufferlib": pufferlib.__file__, "pufferroyale": pufferroyale.__file__}
print(paths)
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "cuda available", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("PD_REF", ref, "visible in container:", os.path.exists(ref))
inside = [k for k, p in paths.items() if os.path.realpath(p).startswith(ref + os.sep)]
inside += [f"sys.path:{p}" for p in sys.path if p and os.path.realpath(p).startswith(ref + os.sep)]
assert not inside, f"a PufferDrive path is in use: {inside}"
assert not os.environ.get("PYTHONPATH"), "PYTHONPATH must be empty (env.sh unsets it)"
assert numpy.__version__.startswith("1."), "numpy must be < 2"
h = hashlib.sha256(open(pufferlib.pufferl.__file__, "rb").read()).hexdigest()
assert h == UPSTREAM_PUFFERL_SHA256, f"pufferl.py is not upstream 3b5c604: {h}"
here = os.path.realpath(os.getcwd())
assert os.path.realpath(pufferroyale.__file__).startswith(here + os.sep), \
    f"pufferroyale comes from {pufferroyale.__file__}, not from the checkout {here}"
print("isolation OK")
