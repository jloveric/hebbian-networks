"""Bounded numerical checks; never launches a dataset training experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from . import common
from . import mnist_connected_groups as mnist
from . import competitive_conv_connected_groups as conv
from . import competitive_conv_connected_sparse as sparse


def copy_state(state):
    return [{key: value.copy() for key, value in bank.items()} for bank in state]


def same_state(actual, expected, *, exact=False):
    for a, b in zip(actual, expected, strict=True):
        for key in ("w", "m", "clock"):
            if exact or key == "clock":
                np.testing.assert_array_equal(a[key], b[key])
            else:
                np.testing.assert_allclose(a[key], b[key], rtol=3e-5, atol=2e-6)


def checkpoint_roundtrip(model, state, config, mean):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        progress = dict(epoch=1, seed=10101, status="verification")
        model.save_checkpoint(path, state, config, mean, progress)
        restored, saved, saved_mean, row = model.load_checkpoint(path)
        assert saved == config and row == progress
        np.testing.assert_array_equal(mean, saved_mean)
        same_state(restored, state, exact=True)


def check_mnist(gpu=False, *, integration="semi_implicit", dropout=0.):
    c = mnist.Config(widths=(8, 8), winners=(4, 4), batch=4,
                     dropout=dropout, integration=integration)
    rng = np.random.default_rng(237)
    x = rng.normal(0, .2, (7, 12)).astype(np.float32)
    labels = np.array([0, 3, 5, 7, 9, 2, 4], np.int64)
    ids = np.array([11, 2, 91, 3, 45, 28, 7], np.int64)
    state = mnist.initialize(c, 10101, inputs=12)
    before = common.fingerprint(state)
    _, energy = mnist.reference_forward(state, x, c)
    assert common.fingerprint(state) == before
    expected = mnist.reference_train(state, x, labels, c, .00075, image_ids=ids, seed=10101, epoch=2)
    reverse = mnist.reference_train(state, x, labels, c, .00075, True, image_ids=ids, seed=10101, epoch=2)
    same_state(expected, reverse, exact=True)
    assert common.fingerprint(expected) != before
    assert common.fingerprint(state) == before
    checkpoint_roundtrip(mnist, expected, c, np.zeros(12, np.float32))
    if gpu:
        actual_energy, _ = mnist.run(state, x, c)
        np.testing.assert_allclose(actual_energy, energy, rtol=3e-5, atol=2e-6)
        assert common.fingerprint(state) == before
        if not dropout:
            changed = copy_state(state)
            mnist.run(changed, x, c, labels, .00075)
            same_state(changed, expected)
    return dict(dataset="mnist", config=asdict(c), images=len(labels), gpu=gpu,
                forward=True, local_updates=True, gpu_update=gpu and not dropout,
                checkpoint_roundtrip=True)


def check_cifar(gpu=False, *, integration="semi_implicit", dropout=0.):
    c = conv.Config(channels=(4, 4), winners=(2, 2), kernels=(3, 3),
                    pool_after=(True, False), batch=2, rate=.00075,
                    dropout_probability=dropout, integration=integration)
    rng = np.random.default_rng(813)
    images = rng.normal(0, .2, (2, 10, 10, 3)).astype(np.float32)
    labels = np.array([2, 8], np.int64)
    state = conv.initialize(c, 10101)
    before = conv.fingerprint(state)
    _, energy = conv.forward(state, images, c)
    masks = conv.mask_plan(c, 10101, 2, len(images)) if dropout else None
    expected = copy_state(state)
    reverse = copy_state(state)
    conv.reference_train(expected, images, labels, c, masks=masks)
    conv.reference_train(reverse, images, labels, c, reverse=True, masks=masks)
    same_state(expected, reverse, exact=True)
    assert conv.fingerprint(expected) != before
    assert conv.fingerprint(state) == before
    checkpoint_roundtrip(conv, expected, c, np.zeros(3, np.float32))
    if gpu:
        packet = sparse.score(state, images, c)
        np.testing.assert_allclose(packet["energies"], energy, rtol=4e-5, atol=2e-6)
        assert conv.fingerprint(state) == before
        if not dropout:
            changed = copy_state(state)
            sparse.train(changed, images, labels, c)
            same_state(changed, expected)
    return dict(dataset="cifar10", config=asdict(c), images=len(labels), gpu=gpu,
                forward=True, local_updates=True, gpu_update=gpu and not dropout,
                checkpoint_roundtrip=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    results = [check(args.gpu, integration=integration)
               for integration in ("semi_implicit", "explicit")
               for check in (check_mnist, check_cifar)]
    results += [check_mnist(args.gpu, dropout=.3), check_cifar(args.gpu, dropout=.2)]
    print(json.dumps(dict(status="passed", results=results), indent=2))


if __name__ == "__main__":
    main()
