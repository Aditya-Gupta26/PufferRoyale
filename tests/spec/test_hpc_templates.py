"""SPEC §15.6 HPC templates (not executed): files exist, sbatch files pass `bash -n`, and every
script path / CLI flag they reference exists (flags checked against the script's --help)."""
import os
import re
import subprocess

import pytest

import helpers as H
import leaguekit as L

HPC = os.path.join(H.ROOT, "hpc")
FILES = ["pufferroyale.def", "train.sbatch", "league.sbatch", "README.md"]
SBATCH = ["train.sbatch", "league.sbatch"]


def read(name):
    with open(os.path.join(HPC, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("name", FILES)
def test_hpc_file_exists(pr, name):
    p = os.path.join(HPC, name)
    assert os.path.isfile(p) and os.path.getsize(p) > 0, f"hpc/{name} missing (SPEC §15.6)"


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_syntax(pr, name):
    out = subprocess.run(["bash", "-n", os.path.join(HPC, name)], capture_output=True, text=True)
    assert out.returncode == 0, f"bash -n hpc/{name}: {out.stderr}"
    assert "#SBATCH" in read(name), f"hpc/{name} has no #SBATCH directives"


def test_apptainer_definition_content(pr):
    d = read("pufferroyale.def")
    assert re.search(r"^\s*Bootstrap\s*:", d, re.M | re.I), "Apptainer definition needs a Bootstrap header"
    assert re.search(r"cuda[^\n]*12|12\.\d[^\n]*cuda", d, re.I), "CUDA 12 base image expected"
    assert "3.12" in d, "Python 3.12 expected"
    assert re.search(r"pufferlib", d, re.I) and "NO_OCEAN" in d, "PufferLib 3.0 from source with the NO_OCEAN fix"


def test_readme_usage(pr):
    r = read("README.md")
    assert "apptainer exec --nv" in r
    assert re.search(r"account", r, re.I) and re.search(r"partition", r, re.I), "account/partition placeholders"
    assert re.search(r"scratch", r, re.I), "scratch paths"


def logical_lines(text):
    text = re.sub(r"\\\r?\n", " ", text)
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def referenced(name):
    """(script, [flags]) for every invocation of scripts/*.py in an sbatch file."""
    refs = []
    for line in logical_lines(read(name)):
        for m in re.finditer(r"(scripts/[A-Za-z0-9_]+\.py)", line):
            tail = line[m.end():]
            tail = re.split(r"[;&|]", tail)[0]
            refs.append((m.group(1), re.findall(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9_.-]*)", tail)))
    return refs


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_scripts_and_flags_exist(pr, name):
    refs = referenced(name)
    assert refs, f"hpc/{name} does not invoke any scripts/*.py"
    problems = []
    cache = {}
    for script, flags in refs:
        if not os.path.isfile(os.path.join(H.ROOT, script)):
            problems.append(f"{script} does not exist")
            continue
        if script not in cache:
            out = L.run_script([script, "--help"], timeout=120)
            cache[script] = (out.returncode, out.stdout + out.stderr)
        rc, text = cache[script]
        if rc != 0:
            problems.append(f"{script} --help failed")
            continue
        advertised = set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_.-]*)", text))
        for f in flags:
            flag = f.split("=")[0]
            if flag in advertised:
                continue
            problems.append(f"{script}: flag {flag} is not in --help")
    assert not problems, f"hpc/{name}: " + "; ".join(problems)
