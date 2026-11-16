from dataclasses import asdict
import json

import numpy as np
import pytest

from hebbian_networks import common
from hebbian_networks import mnist_connected_groups as mnist
from hebbian_networks import competitive_conv_connected_sparse as cifar
from hebbian_networks.scoring import score_breakdown
from hebbian_networks.train import evaluate_final, recipe, timestep
from hebbian_networks.verify import check_mnist, check_cifar


def test_mnist_reference_and_roundtrip():
    check_mnist()


def test_cifar_reference_and_roundtrip():
    check_cifar()


def test_counter_dropout_preserves_example_identity_across_chunks():
    ids = np.array([18, 3, 71, 4, 92], dtype=np.int64)
    whole = common.dropout_mask(ids, 100, 10102, 42, 1, .3)
    chunks = np.concatenate([common.dropout_mask(part, 100, 10102, 42, 1, .3)
                             for part in (ids[:2], ids[2:])])
    np.testing.assert_array_equal(whole, chunks)
    assert not np.array_equal(whole, common.dropout_mask(ids, 100, 10102, 43, 1, .3))


def test_recipes_match_recorded_paper_evidence():
    evidence = json.loads((common.REPO / "results/paper.json").read_text())
    for dataset, name, model in (("mnist", "MNIST", mnist), ("cifar10", "CIFAR-10", cifar)):
        settings = recipe(dataset)
        recorded = next(r for r in evidence["results"] if r["dataset"] == name)
        assert settings["seed"] == recorded["seed"]
        assert asdict(model.Config(**settings["config"])) == asdict(model.Config(**recorded["config"]))
        assert settings["training_images"] == recorded["training_images"]
        assert settings["epochs"] == 100


def test_schedule_and_fixed_layer_scoring():
    assert timestep(.00075, 0) == timestep(.00075, 19) == timestep(.00075, 20) == .00075
    assert 0 < timestep(.00075, 99) < .00075
    assert timestep(.00075, 100) == 0
    labels = np.array([0, 1])
    energy = np.zeros((2, 10, 3), dtype=np.float32)
    energy[0, 0, :] = [3, 4, 5]
    energy[1, 1, :] = [4, 5, 6]
    result, individual, prefixes = score_breakdown(energy, energy / np.array([1, 2, 3], np.float32), labels)
    assert all(row["errors"] == 0 for row in result["individual"] + result["prefix"])
    np.testing.assert_array_equal(individual, np.broadcast_to(labels[:, None], (2, 3)))
    np.testing.assert_array_equal(prefixes, individual)


def test_official_test_is_not_loaded_for_incomplete_checkpoint(tmp_path, monkeypatch):
    config = mnist.Config(widths=(4, 4), winners=(2, 2))
    state = mnist.initialize(config, 1, inputs=4)
    path = tmp_path / "incomplete.npz"
    mnist.save_checkpoint(path, state, config, np.zeros(4, np.float32),
                          dict(epoch=99, status="training", seed=1, fresh_weights=True))
    from hebbian_networks import data
    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete checkpoint must not read test data")
    monkeypatch.setattr(data, "mnist_test", forbidden)
    with pytest.raises(ValueError, match="epoch-100"):
        evaluate_final("mnist", path, tmp_path)
