"""Build the pufferroyale.binding extension (PufferLib 3.0 native env + debug API).

    python setup.py build_ext --inplace        # release (-O2)
    DEBUG=1 python setup.py build_ext --inplace --force   # -O0 -g + ASan/UBSan
    uv pip install -e .            (or pip install -e .)       engine + env API
    uv pip install -e '.[train]'   (or pip install -e '.[train]') + torch / PufferLib for training

The engine is header-only C99 under pufferroyale/csrc/; the only translation unit is
pufferroyale/binding.c, compiled against the vendored pufferroyale/env_binding.h.

raylib is optional and only linked with PR_RAYLIB=1: from $RAYLIB_DIR if set, else from
third_party/raylib-5.5_<platform> (fetch it with tools/get_raylib.sh). The directory must hold
include/raylib.h and lib/libraylib.a.

PufferLib 3.0 is not a plain PyPI dependency here: install it from source with NO_OCEAN=1 (see
README.md, "Install and build", and docs/DECISIONS.md D6).
"""
import os
import sys

import numpy
from setuptools import Extension, find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.environ.get("DEBUG", "0") == "1"
RAYLIB = os.environ.get("PR_RAYLIB", "0") == "1"


def here(*parts):
    """Absolute path inside the project, whatever the current directory is."""
    return os.path.join(HERE, *parts)


extra_compile_args = ["-std=c99", "-Wall", "-Wextra"]
extra_link_args = []
define_macros = []
include_dirs = [numpy.get_include(), here("pufferroyale"), here("pufferroyale", "csrc")]
libraries = []
library_dirs = []

if DEBUG:
    extra_compile_args += ["-O0", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined"]
    extra_link_args += ["-fsanitize=address,undefined"]
    define_macros += [("PR_DEBUG", "1")]
else:
    extra_compile_args += ["-O2"]

if RAYLIB:
    default = "raylib-5.5_macos" if sys.platform == "darwin" else "raylib-5.5_linux_amd64"
    rl = os.path.abspath(os.environ.get("RAYLIB_DIR") or here("third_party", default))
    lib = os.path.join(rl, "lib", "libraylib.a")
    if not (os.path.exists(os.path.join(rl, "include", "raylib.h")) and os.path.exists(lib)):
        sys.exit(f"PR_RAYLIB=1 but no raylib at {rl} (want include/raylib.h and lib/libraylib.a): "
                 "run tools/get_raylib.sh or set RAYLIB_DIR")
    include_dirs.append(os.path.join(rl, "include"))
    define_macros.append(("PR_RAYLIB", "1"))
    extra_link_args += [lib]
    if sys.platform == "darwin":
        extra_link_args += ["-framework", "Cocoa", "-framework", "IOKit", "-framework", "OpenGL",
                            "-framework", "CoreVideo", "-framework", "CoreAudio", "-framework", "CoreGraphics"]
    else:
        libraries += ["GL", "m", "pthread", "dl", "rt", "X11"]

CSRC = here("pufferroyale", "csrc")
ext = Extension(
    "pufferroyale.binding",
    sources=[here("pufferroyale", "binding.c")],
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
    library_dirs=library_dirs,
    depends=[os.path.join(CSRC, f) for f in sorted(os.listdir(CSRC)) if f.endswith(".h")]
    + [here("pufferroyale", "env_binding.h")],
)

setup(
    name="pufferroyale",
    version="0.1.1",
    description="PufferRoyale: a deterministic integer Clash Royale 1v1 battle simulator (PufferLib 3.0 env)",
    packages=find_packages(where=HERE, include=["pufferroyale", "pufferroyale.*"]),
    package_data={"pufferroyale": ["csrc/*.h", "env_binding.h", "config/*.ini", "PUFFERLIB_LICENSE"]},
    ext_modules=[ext],
    python_requires=">=3.10",
    install_requires=["numpy", "gymnasium"],
    # PufferLib 3.0 comes from source (NO_OCEAN=1, see README); torch is PuffeRL's backend
    extras_require={"train": ["torch", "pufferlib>=3.0,<3.1"]},
    zip_safe=False,
)
