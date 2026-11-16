"""Shared numerical, serialization and standalone Mojo build utilities."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
KERNELS = Path(__file__).resolve().parent / "kernels"
BUILD = Path(os.environ.get("HEBBIAN_BUILD_DIR", REPO / ".cache/build"))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def fingerprint(state):
    h = hashlib.sha256()
    for bank in state:
        for key in sorted(bank):
            a = np.asarray(bank[key])
            h.update(str((key, a.shape, a.dtype)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def dropout_mask(image_ids, dimension, seed, epoch, layer, probability):
    """Counter-based masks, shared across class columns and independent of chunks."""
    key = np.asarray(image_ids, np.uint32)[:, None] ^ np.uint32((seed*0x9E3779B9) & 0xFFFFFFFF)
    key ^= np.uint32((epoch*0x85EBCA6B) & 0xFFFFFFFF)
    key ^= np.uint32((layer*0xC2B2AE35) & 0xFFFFFFFF)
    key = key ^ (np.arange(dimension, dtype=np.uint32)[None, :]*np.uint32(0x27D4EB2D))
    key = (key ^ (key >> 16))*np.uint32(0x7FEB352D)
    key = (key ^ (key >> 15))*np.uint32(0x846CA68B)
    key ^= key >> 16
    return (key >> 8).astype(np.float32)/np.float32(16777216) >= np.float32(probability)


def preprocess(x, mean, standard_deviation=None):
    centered = x-mean
    if standard_deviation is not None:
        std = np.asarray(standard_deviation, np.float32)
        if std.shape != mean.shape or not np.isfinite(std).all() or (std < .1).any():
            raise ValueError("Invalid saved training normalization statistics")
        centered = centered/std
    maximum = np.abs(centered).max(1, keepdims=True)
    return np.divide(centered, maximum, out=np.zeros_like(centered), where=maximum > 0)


def balanced(labels, count, seed):
    if count == len(labels):
        return np.arange(count)
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels == c), count//10, replace=False)
                             for c in range(10)])
    return rng.permutation(chosen)


def source_paths(source, python_paths=()):
    paths = set(python_paths)
    pending = [source]
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        for name in re.findall(r"^from ([A-Za-z0-9_]+) import", path.read_text(), re.M):
            child = KERNELS / (name + ".mojo")
            if child.exists() and child not in paths:
                pending.append(child)
    return sorted(paths)


def backend(module_name, source, paths):
    compiler = os.environ.get("MOJO_BIN") or shutil.which("mojo")
    if not compiler:
        raise RuntimeError("Mojo 1.2.0 is required: install the gpu extra or set MOJO_BIN")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env.setdefault("MODULAR_CACHE_DIR", str(BUILD / "modular"))
    version = subprocess.check_output([compiler, "--version"], env=env, text=True).strip()
    if not version.startswith("Mojo 1.2.0 "):
        raise RuntimeError(f"Expected Mojo 1.2.0, found {version}")
    sources = {str(p.relative_to(REPO)): digest(p) for p in sorted(set(paths) | {Path(__file__)})}
    identity = hashlib.sha256(json.dumps(dict(sources=sources, compiler=version,
        python=sys.version, numpy=np.__version__), sort_keys=True).encode()).hexdigest()[:20]
    directory = BUILD / module_name / identity
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / (module_name + ".so")
    # Serialize builds of an identical extension across independent Python workers.
    import fcntl
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            temporary = binary.with_suffix(".tmp.so")
            subprocess.run([compiler, "build", str(source), "-I", str(KERNELS),
                "--emit", "shared-lib", "--fp-mode", "contract=off", "-o", str(temporary)],
                env=env, check=True)
            temporary.replace(binary)
    sys.path.insert(0, str(directory))
    return importlib.import_module(module_name), dict(compiler=version, sources=sources,
        binary=str(binary), binary_sha256=digest(binary))
