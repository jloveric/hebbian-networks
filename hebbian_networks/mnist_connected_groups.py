"""A dense MLP whose receiving neurons own class-specific local updates."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from . import common
from . import mnist_reference as pooled

ROOT = common.BUILD / "mnist"
SOURCE = common.KERNELS / "mnist_connected_groups.mojo"
VERSION = "mnist-connected-class-groups-1"


@dataclass(frozen=True)
class Config:
    widths: tuple = (64, 64)  # Neurons per class; all ten groups connect onward.
    winners: tuple = (4, 4)
    batch: int = 32
    margin: float = .10
    dropout: float = 0.
    direction_mix: tuple = (.125, .875)
    integration: str = "semi_implicit"

    def __post_init__(self):
        for name in ("widths", "winners", "direction_mix"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.integration not in ("semi_implicit", "explicit")
                or not 2 <= len(self.widths) <= 4 or len(self.widths) != len(self.winners)
                or any(type(w) is not int or type(k) is not int or not 1 <= k <= w <= 512
                       for w, k in zip(self.widths, self.winners))
                or type(self.batch) is not int or not 1 <= self.batch <= 256
                or not np.isfinite(self.margin) or self.margin < 0
                or not np.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or len(self.direction_mix) != 2 or not np.isfinite(self.direction_mix).all()
                or any(v < 0 or v > 1 for v in self.direction_mix)
                or not np.isclose(sum(self.direction_mix), 1., rtol=0, atol=1e-7)):
            raise ValueError("invalid connected group configuration")


def initialize(c=Config(), seed=10101, inputs=784):
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, k])) for k in range(10)]
    state = []
    for width in c.widths:
        weights = []
        for rng in rngs:
            w = rng.normal(size=(width, inputs))
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            weights.append(np.column_stack((w, np.zeros(width))).astype(np.float32))
        w = np.stack(weights)
        state.append(dict(w=w, m=np.zeros_like(w), clock=np.zeros((10, width), np.int64)))
        inputs = 10 * width
    return state


def scores(energy, c):
    return (energy / np.asarray(c.winners, np.float32)).sum(-1)


def reference_forward(state, images, c, *, training=False, image_ids=None, seed=10101, epoch=0):
    """Label-blind NumPy forward; no calls to independent-column forwards."""
    features = np.asarray(images, np.float32)
    ids = np.arange(len(features)) if image_ids is None else np.asarray(image_ids)
    records, energies = [], []
    for layer, bank in enumerate(state):
        incoming = np.column_stack((features, np.ones(len(features), np.float32)))
        if layer == 0:
            norms = np.sqrt(np.sum(incoming.astype(np.float64)**2, axis=1, keepdims=True))
            incoming = (incoming / norms).astype(np.float32)
        elif training and c.dropout:
            keep = common.dropout_mask(ids, incoming.shape[1]-1, seed, epoch, layer, c.dropout)
            incoming[:, :-1] = np.where(keep, incoming[:, :-1]/(np.float32(1)-np.float32(c.dropout)), 0)
        u = np.broadcast_to(incoming[:, None, :], (len(features), 10, incoming.shape[1]))
        z = np.einsum("bi,cji->bcj", incoming, bank["w"])
        selected_ids = np.argsort(-np.abs(z), axis=-1, kind="stable")[..., :c.winners[layer]]
        h = np.zeros_like(z)
        np.put_along_axis(h, selected_ids, np.take_along_axis(z, selected_ids, -1), -1)
        records.append((u, h))
        energies.append(np.sum(h.astype(np.float64)**2, -1).astype(np.float32))
        features = h.reshape(len(h), -1)
    return records, np.stack(energies, -1)


def pair_signs(energy, labels, c):
    score = scores(energy, c)
    other = score.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    rival = other.argmax(1)
    truth = score[np.arange(len(labels)), labels]
    wrong = other.max(1)
    learn = (score.argmax(1) != labels) | ((truth-wrong)/(truth+wrong+1e-30) < c.margin)
    signs = np.zeros(score.shape, np.float32)
    at = np.flatnonzero(learn)
    signs[at, labels[at]] = 1
    signs[at, rival[at]] = -1
    return signs


def reference_train(state, images, labels, c, rate, reverse=False, *, image_ids=None, seed=10101, epoch=0):
    result = [{k: a.copy() for k, a in bank.items()} for bank in state]
    ids = np.arange(len(images)) if image_ids is None else np.asarray(image_ids)
    for start in range(0, len(images), c.batch):
        records, energy = reference_forward(result, images[start:start+c.batch], c, training=True,
                                           image_ids=ids[start:start+c.batch], seed=seed, epoch=epoch)
        signs = pair_signs(energy, labels[start:start+c.batch], c)
        # Every observation is saved before the first commit.
        pending = [pooled.statistics(u, h, signs) for u, h in records]
        order = reversed(range(len(result))) if reverse else range(len(result))
        for layer in order:
            pooled.commit(result[layer], *pending[layer], rate,
                          direction_mix=c.direction_mix, integration=c.integration)
    return result


def source_paths():
    return common.source_paths(SOURCE, [Path(__file__), Path(common.__file__), Path(pooled.__file__)])


@lru_cache(None)
def backend():
    return common.backend("mnist_connected_groups_native", SOURCE, source_paths())


def validate(state, x, c):
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or len(state) != len(c.widths):
        raise ValueError("invalid images or layer count")
    incoming = x.shape[1]
    for bank, width in zip(state, c.widths):
        for key in ("w", "m"):
            a = bank[key]
            if (a.shape != (10, width, incoming+1) or a.dtype != np.float32
                    or not a.flags.c_contiguous or not np.isfinite(a).all()):
                raise ValueError("invalid connected weight/history geometry")
        a = bank["clock"]
        if a.shape != (10, width) or a.dtype != np.int64 or not a.flags.c_contiguous or (a < 0).any():
            raise ValueError("invalid optimizer clocks")
        incoming = 10*width


def run(state, images, c, labels=None, rate=.00075, reverse=False, observations=None,
        *, image_ids=None, seed=10101, epoch=0):
    x = np.ascontiguousarray(images, np.float32)
    validate(state, x, c)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("invalid timestep")
    y = np.zeros(len(x), np.int64) if labels is None else np.ascontiguousarray(labels, np.int64)
    if y.shape != (len(x),) or ((y < 0) | (y >= 10)).any():
        raise ValueError("invalid labels")
    ids = np.arange(len(x), dtype=np.int64) if image_ids is None else np.ascontiguousarray(image_ids, np.int64)
    if (ids.shape != (len(x),) or ((ids < 0) | (ids > 0xFFFFFFFF)).any()
            or not 0 <= seed <= 0xFFFFFFFF or not 0 <= epoch <= 0xFFFFFFFF):
        raise ValueError("invalid deterministic mask counters")
    if observations is not None and len(x) > c.batch:
        raise ValueError("capture requires a single minibatch")
    captured = [] if observations is None else [
        dict(u=np.zeros((10, len(x), b["w"].shape[-1]), np.float32),
             h=np.zeros((10, len(x), w), np.float32)) for b, w in zip(state, c.widths)]
    packet = dict(config=asdict(c), state=state, images=x, labels=y, train=labels is not None,
                  image_ids=ids, seed=seed, epoch=epoch,
                  rate=rate, reverse=reverse, energies=np.zeros((len(x), 10, len(state)), np.float32),
                  events=np.zeros((len(x), 2), np.float32), observations=captured,
                  usage=[np.zeros((10, w), np.int64) for w in c.widths])
    native, _ = backend()
    native.run(packet)
    if not np.isfinite(packet["energies"]).all() or any(
            not np.isfinite(b[k]).all() for b in state for k in ("w", "m")):
        raise FloatingPointError("nonfinite connected model state or activity")
    if observations is not None:
        observations.extend((b["u"].transpose(1, 0, 2), b["h"].transpose(1, 0, 2)) for b in captured)
    counts = dict(images=len(x), included=int(packet["events"][:, 0].sum()),
                  mistakes=int(packet["events"][:, 1].sum()), usage=packet["usage"])
    return packet["energies"], counts


def save_checkpoint(path, state, c, mean, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(version=VERSION, config=asdict(c), progress=progress, fingerprint=common.fingerprint(state))
    payload = {f"layer{i}_{key}": a for i, bank in enumerate(state) for key, a in bank.items()}
    temp = path.with_suffix(".tmp.npz")
    with temp.open("wb") as output:
        np.savez(output, metadata=json.dumps(meta), mean=mean, **payload)
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"].item()))
        if meta["version"] != VERSION:
            raise ValueError("unsupported connected checkpoint")
        c = Config(**meta["config"])
        state = [{key: saved[f"layer{i}_{key}"].copy() for key in ("w", "m", "clock")}
                 for i in range(len(c.widths))]
        mean = saved["mean"].copy()
    if common.fingerprint(state) != meta["fingerprint"]:
        raise ValueError("checkpoint state mismatch")
    validate(state, np.zeros((1, state[0]["w"].shape[-1]-1), np.float32), c)
    return state, c, mean, meta["progress"]
