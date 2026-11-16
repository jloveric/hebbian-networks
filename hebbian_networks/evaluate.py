"""Evaluate an existing final checkpoint, optionally including layer scores."""
import argparse
from pathlib import Path
import json

import numpy as np

from . import common, data
from .scoring import score_breakdown
from .train import evaluate_final, model_for, state_hash


def evaluate(dataset, checkpoint, data_directory, layers=False):
    result = evaluate_final(dataset, checkpoint, data_directory)
    if layers:
        model, _ = model_for(dataset)
        state, config, mean, _ = model.load_checkpoint(checkpoint)
        before = state_hash(dataset, state)
        ds = data.mnist_test(data_directory, mean) if dataset == "mnist" else data.cifar_test(data_directory, mean)
        if dataset == "mnist":
            energy, _ = model.run(state, ds["images"], config)
            normalized = energy / np.asarray(config.winners, np.float32)
        else:
            energy = model.score(state, ds["images"], config, evaluation_batch=128)["energies"]
            normalized = energy * model.score_weights(config)[None, None, :]
        breakdown, _, _ = score_breakdown(energy, normalized, ds["labels"])
        assert state_hash(dataset, state) == before
        assert common.digest(checkpoint) == result["checkpoint_sha256"]
        assert breakdown["prefix"][-1]["errors"] == result["metrics"]["errors"]
        result["layer_scores"] = breakdown
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("mnist", "cifar10"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--layers", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.checkpoint,
                      args.data_dir or common.REPO / "data" / args.dataset, args.layers)
    if args.output is not None:
        data.save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
