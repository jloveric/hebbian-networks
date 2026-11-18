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


@pytest.mark.parametrize(
    "integration,dropout", [("semi_implicit", 0.), ("explicit", 0.), ("semi_implicit", .3)]
)
def test_mnist_reference_and_roundtrip(integration, dropout):
    check_mnist(integration=integration, dropout=dropout)


@pytest.mark.parametrize(
    "integration,dropout", [("semi_implicit", 0.), ("explicit", 0.), ("semi_implicit", .2)]
)
def test_cifar_reference_and_roundtrip(integration, dropout):
    check_cifar(integration=integration, dropout=dropout)


def test_packaged_kernels_belong_to_the_paper_backends():
    dependencies = {path for path in mnist.source_paths() + cifar.source_paths()
                    if path.suffix == ".mojo"}
    assert dependencies == set(common.KERNELS.glob("*.mojo"))


def test_paper_gpu_training_requires_zero_dropout():
    with pytest.raises(ValueError, match="dropout=0"):
        mnist.run([], np.zeros((1, 784), np.float32), mnist.Config(dropout=.3), np.array([0]))
    with pytest.raises(ValueError, match="dropout_probability=0"):
        cifar.train([], np.zeros((1, 32, 32, 3), np.float32), np.array([0]),
                    cifar.Config(dropout_probability=.2))


def test_counter_dropout_preserves_example_identity_across_chunks():
    ids = np.array([18, 3, 71, 4, 92], dtype=np.int64)
    whole = common.dropout_mask(ids, 100, 10102, 42, 1, .3)
    chunks = np.concatenate([common.dropout_mask(part, 100, 10102, 42, 1, .3)
                             for part in (ids[:2], ids[2:])])
    np.testing.assert_array_equal(whole, chunks)
    assert not np.array_equal(whole, common.dropout_mask(ids, 100, 10102, 43, 1, .3))


def test_recipes_match_recorded_paper_evidence():
    results_dir = common.REPO / "results"
    evidence = json.loads((results_dir / "paper.json").read_text())
    for dataset, model, dropout_key in (("mnist", mnist, "dropout"),
                                        ("cifar10", cifar, "dropout_probability")):
        initial_hashes = {}
        for integration in ("semi_implicit", "explicit"):
            settings = recipe(dataset, integration)
            recorded = next(r for r in evidence["results"]
                            if r["dataset"] == dataset and r["integration"] == integration)
            assert settings["config"][dropout_key] == 0
            assert asdict(model.Config(**settings["config"])) == asdict(model.Config(**recorded["config"]))
            assert settings["training_images"] == recorded["training_images"]
            assert settings["epochs"] == recorded["epochs"] == 100
            assert settings["seed"] == settings["paper_seeds"][0]
            assert settings["paper_seeds"] == [r["seed"] for r in recorded["runs"]]
            for run in recorded["runs"]:
                path = results_dir / run["record"]
                assert common.digest(path) == run["record_sha256"]
                result = json.loads(path.read_text())
                metrics = result["test" if dataset == "mnist" else "metrics"]
                assert result["seed"] == run["seed"] and result["epoch"] == settings["epochs"]
                assert metrics["images"] == settings["test_images"]
                assert metrics["errors"] == run["errors"]
                initial_hash = run["initial_state_sha256"]
                assert initial_hashes.setdefault(run["seed"], initial_hash) == initial_hash


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
