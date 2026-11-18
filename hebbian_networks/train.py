"""Train a dropout-free paper recipe, resume, and test the final model.

Training mathematics, RNG streams and epoch boundaries match the source runs.
"""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from . import common, data
from . import mnist_connected_groups as mnist
from . import competitive_conv_connected_sparse as cifar
from .metrics import mnist_metrics, cifar_metrics

MEASURE = (0, 1, 5, 10, 20, 40, 60, 80, 100)
INTEGRATIONS = ("semi_implicit", "explicit")
# Original CIFAR-10 identity hashes the full training source, indices, mean and sample.
CIFAR_FULL_DATA_IDENTITY = "e6714d4bede633bad52b3f61f724c31ff2390df029ca8860afd8144e47ebb0cb"
STOP = False


def timestep(initial, completed, dataset="cifar10"):
    """The source schedule uses completed epochs, including its epoch-21 plateau."""
    if not 0 <= completed <= 100:
        raise ValueError("Invalid schedule position")
    if completed <= 20:
        return initial
    if dataset == "mnist":
        # Keep the original MNIST runner's floating-point operation order.
        return initial * (1 + math.cos(math.pi * (completed - 20) / (100 - 20))) / 2
    return initial * .5 * (1 + math.cos(math.pi * (completed - 20) / 80))


def recipe(dataset, integration=None):
    directory = common.REPO / "configs"
    if not directory.exists():  # Packaged wheel rather than an editable checkout.
        directory = Path(__file__).parent / "configs"
    settings = json.loads((directory / (dataset + ".json")).read_text())
    if integration is not None:
        if integration not in INTEGRATIONS:
            raise ValueError(f"Unknown Lion update: {integration}")
        settings["config"]["integration"] = integration
    return settings


def model_for(dataset):
    if dataset == "mnist":
        return mnist, mnist_metrics
    if dataset == "cifar10":
        return cifar, cifar_metrics
    raise ValueError(dataset)


def state_hash(dataset, state):
    return common.fingerprint(state) if dataset == "mnist" else cifar.fingerprint(state)


def stop(signum, frame):
    global STOP
    STOP = True


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(common.KERNELS.glob("*.mojo"))
    return {str(p.relative_to(common.REPO)): common.digest(p) for p in paths}


def evaluate_final(dataset, checkpoint, data_directory, output=None):
    """Test only a completed epoch-100 full-data checkpoint; never fit anything."""
    model, metrics = model_for(dataset)
    checkpoint = Path(checkpoint)
    state, config, mean, progress = model.load_checkpoint(checkpoint)
    expected_images = 60000 if dataset == "mnist" else 50000
    if (progress["epoch"] != 100 or progress["status"] != "complete"
            or not progress["fresh_weights"]):
        raise ValueError("Official testing requires a complete, freshly trained epoch-100 model")
    provenance = progress["provenance"]
    trained_images = provenance.get("training_images", provenance.get("images"))
    if (trained_images is None and dataset == "cifar10"
            and provenance.get("version") in ("paper-ablations-1", "paper-additional-seeds-1")
            and provenance.get("data_identity") == CIFAR_FULL_DATA_IDENTITY):
        trained_images = expected_images
    if trained_images != expected_images or progress["measurements"][-1]["training"]["images"] != expected_images:
        raise ValueError("Checkpoint was not trained on the complete official training set")
    before = state_hash(dataset, state)
    checksum = common.digest(checkpoint)
    if output is not None and Path(output).exists():
        cached = json.loads(Path(output).read_text())
        if cached["checkpoint_sha256"] != checksum or cached["dataset"] != dataset:
            raise ValueError("Existing evaluation belongs to a different checkpoint")
        return cached
    ds = data.mnist_test(data_directory, mean) if dataset == "mnist" else data.cifar_test(data_directory, mean)
    if len(ds["labels"]) != 10000:
        raise ValueError("Complete official test partition required")
    started = time.perf_counter()
    measured = metrics(state, ds["images"], ds["labels"], config)
    if state_hash(dataset, state) != before or common.digest(checkpoint) != checksum:
        raise RuntimeError("Evaluation changed the checkpoint or persistent state")
    result = dict(dataset=dataset, seed=progress["seed"], integration=config.integration,
                  epoch=100, metrics=measured,
                  checkpoint_sha256=checksum, state_sha256=before, test_sources=ds["sources"],
                  seconds=time.perf_counter()-started, evaluation_preserved_state=True,
                  official_test_used_for_selection=False, prior_validation_and_test_reuse=True)
    if output is not None:
        data.save_json(output, result)
    return result


def train(dataset, output, data_directory, seed=None, *, integration=None):
    settings = recipe(dataset, integration)
    seed = settings["seed"] if seed is None else seed
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Seed must fit an unsigned 32-bit integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("w") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(dataset, output, Path(data_directory), seed, settings)


def _train(dataset, output, data_directory, seed, settings):
    model, metrics = model_for(dataset)
    config = model.Config(**settings["config"])
    ds = data.mnist_training(data_directory) if dataset == "mnist" else data.cifar_training(data_directory)
    if len(ds["labels"]) != settings["training_images"]:
        raise ValueError("Complete official training partition required")
    if dataset == "mnist":
        ds["sample"] = common.balanced(ds["labels"], 4000, 7101)
    _, backend = model.backend()
    provenance = dict(version="paper-standalone-1", dataset=dataset, seed=seed,
        config=asdict(config), schedule=settings["schedule"], initial_rate=settings["initial_rate"],
        training_images=len(ds["labels"]), training_sources=ds["sources"],
        mean_sha256=common.array_hash(ds["mean"]), sources=source_hashes(), backend=backend,
        augmentation=False, weight_averaging=False, official_test_used_for_selection=False)
    provenance = json.loads(json.dumps(provenance))
    method = output / "method.json"
    if method.exists() and json.loads(method.read_text()) != provenance:
        raise ValueError("Recipe, sources, build or data changed since this run was created")
    data.save_json(method, provenance)
    resume = output / "resume.npz"
    if resume.exists():
        state, saved, mean, progress = model.load_checkpoint(resume)
        if saved != config or progress["provenance"] != provenance or progress["seed"] != seed:
            raise ValueError("Resume does not match this run")
        np.testing.assert_array_equal(mean, ds["mean"])
        if progress["next_rate"] != timestep(settings["initial_rate"], progress["epoch"], dataset):
            raise ValueError("Resume schedule changed")
        if progress["status"] == "interrupted":
            progress["status"] = "training"
    else:
        state = model.initialize(config, seed)
        progress = dict(seed=seed, epoch=0, status="training", fresh_weights=True,
            provenance=provenance, initial_state_sha256=state_hash(dataset, state),
            measurements=[], training_seconds=0., included_images=0, online_mistakes=0,
            consecutive_all_zero=0)

    def persist():
        progress["next_rate"] = timestep(settings["initial_rate"], progress["epoch"], dataset)
        model.save_checkpoint(resume, state, config, ds["mean"], progress)
        data.save_json(output / "trajectory.json", progress)

    def measure(epoch, rate=None, seconds=0.):
        ids = np.arange(len(ds["labels"])) if epoch == 100 else ds["sample"]
        measured = metrics(state, ds["images"][ids], ds["labels"][ids], config)
        progress["measurements"].append(dict(epoch=epoch, rate=rate, training=measured,
                                              training_epoch_seconds=seconds))
        progress["consecutive_all_zero"] = progress["consecutive_all_zero"] + 1 if measured["all_zero_fraction"] > .99 else 0
        if progress["consecutive_all_zero"] >= 2:
            progress["status"] = "all_zero"
        print(json.dumps(dict(dataset=dataset, epoch=epoch, training=measured)), flush=True)

    if not resume.exists():
        measure(0)
        persist()
    for epoch in range(progress["epoch"] + 1, 101):
        if progress["status"] != "training":
            break
        if STOP:
            progress["status"] = "interrupted"
            persist()
            return progress
        order = np.random.default_rng(seed * 1000 + epoch - 1).permutation(len(ds["labels"]))
        rate = timestep(settings["initial_rate"], epoch - 1, dataset)
        started = time.perf_counter()
        try:
            if dataset == "mnist":
                _, events = model.run(state, ds["images"][order], config, ds["labels"][order],
                    rate=rate)
                progress["included_images"] += events["included"]
                progress["online_mistakes"] += events["mistakes"]
            else:
                packet = model.train(state, ds["images"][order], ds["labels"][order],
                    replace(config, rate=rate))
                progress["included_images"] += int(np.any(packet["scales"] != 0, axis=1).sum())
        except FloatingPointError as error:
            # Keep the preceding valid resume; a partially changed failed epoch
            # is never mislabeled as an epoch-boundary checkpoint.
            data.save_json(output / "failure.json", dict(epoch=epoch, reason=str(error)))
            raise
        elapsed = time.perf_counter() - started
        progress["training_seconds"] += elapsed
        progress["epoch"] = epoch
        if epoch in MEASURE:
            measure(epoch, rate, elapsed)
        if epoch == 100 and progress["status"] == "training":
            progress["status"] = "complete"
        persist()
    if progress["status"] == "complete":
        final = output / "final.npz"
        if final.exists():
            if common.digest(final) != common.digest(resume):
                raise ValueError("Existing final checkpoint differs from the completed resume")
        else:
            import shutil
            shutil.copyfile(resume, final)
        if not STOP:
            result = evaluate_final(dataset, final, data_directory, output / "official-test.json")
            print(json.dumps(result), flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("mnist", "cifar10"))
    parser.add_argument("--data-dir", type=Path, help="Official dataset cache directory")
    parser.add_argument("--output", type=Path, help="Training output directory")
    parser.add_argument("--seed", type=int, help="Defaults to the original seed in the four-seed comparison")
    parser.add_argument("--integration", choices=INTEGRATIONS, help="Lion update (default: semi_implicit)")
    parser.add_argument("--dry-run", action="store_true", help="Print the fixed recipe; no data, build or GPU access")
    args = parser.parse_args()
    settings = recipe(args.dataset, args.integration)
    if args.dry_run:
        if args.seed is not None:
            settings["seed"] = args.seed
        print(json.dumps(settings, indent=2))
        return
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    seed = settings["seed"] if args.seed is None else args.seed
    integration = settings["config"]["integration"]
    default_output = common.REPO / "runs" / args.dataset / integration / f"seed{seed}"
    train(args.dataset, args.output or default_output,
          args.data_dir or common.REPO / "data" / args.dataset, seed, integration=integration)


if __name__ == "__main__":
    main()
