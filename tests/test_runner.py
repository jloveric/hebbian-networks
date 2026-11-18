"""Exercise epoch-boundary resume with real local updates on tiny synthetic data."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from hebbian_networks import mnist_connected_groups as model
from hebbian_networks import train as runner
from hebbian_networks.verify import same_state


def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    config = model.Config(widths=(4, 4), winners=(2, 2), batch=4, integration="explicit")
    rng = np.random.default_rng(28)
    images = rng.normal(0, .2, (20, 12)).astype(np.float32)
    labels = np.tile(np.arange(10), 2)
    ds = dict(images=images, labels=labels, mean=np.zeros(12, np.float32), sources={"synthetic": "fixed"})
    settings = dict(seed=10102, config=asdict(config), training_images=20,
                    initial_rate=.00075, schedule="constant20_cosine80")
    control = dict(interrupt=False)

    def run(state, x, c, y, rate, *, image_ids, seed, epoch):
        changed = model.reference_train(state, x, y, c, rate, image_ids=image_ids, seed=seed, epoch=epoch)
        for actual, new in zip(state, changed):
            for key in actual:
                actual[key][...] = new[key]
        if control["interrupt"] and epoch == 1:
            runner.STOP = True
        return None, dict(included=0, mistakes=0)

    def metric(state, x, y, c):
        _, energy = model.reference_forward(state, x, c)
        errors = int((model.scores(energy, c).argmax(1) != y).sum())
        return dict(images=len(y), errors=errors, accuracy=1-errors/len(y), all_zero_fraction=0.)

    cpu = SimpleNamespace(Config=model.Config, initialize=lambda c, seed: model.initialize(c, seed, inputs=12),
        backend=lambda: (None, {}), load_checkpoint=model.load_checkpoint,
        save_checkpoint=model.save_checkpoint, run=run)
    monkeypatch.setattr(runner, "recipe", lambda dataset, integration: settings)
    monkeypatch.setattr(runner, "model_for", lambda dataset: (cpu, metric))
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    monkeypatch.setattr(runner.data, "mnist_training", lambda directory: dict(ds))
    monkeypatch.setattr(runner.common, "balanced", lambda y, count, seed: np.arange(len(y)))
    monkeypatch.setattr(runner, "STOP", False)
    tested = []

    def official_test(dataset, checkpoint, directory, output):
        _, _, _, progress = model.load_checkpoint(checkpoint)
        assert progress["epoch"] == 100 and progress["status"] == "complete"
        tested.append(checkpoint)
        return dict(complete=True)

    monkeypatch.setattr(runner, "evaluate_final", official_test)
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    runner.train("mnist", continuous, tmp_path)
    assert len(tested) == 1
    control["interrupt"] = True
    interrupted = runner.train("mnist", resumed, tmp_path)
    assert interrupted["epoch"] == 1 and interrupted["status"] == "interrupted"
    assert len(tested) == 1 and not (resumed / "final.npz").exists()
    control["interrupt"] = False
    runner.STOP = False
    runner.train("mnist", resumed, tmp_path)
    assert len(tested) == 2
    a, ca, ma, pa = model.load_checkpoint(continuous / "final.npz")
    b, cb, mb, pb = model.load_checkpoint(resumed / "final.npz")
    same_state(a, b, exact=True)
    assert ca == cb and pa["next_rate"] == pb["next_rate"] == 0
    np.testing.assert_array_equal(ma, mb)
