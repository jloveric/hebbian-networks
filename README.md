# Supervised Hebbian Class Networks

Standalone code for the jointly trained MNIST MLP and CIFAR-10 convolutional
network reported in the TMLR draft. Every neuron or filter is assigned a class,
and all layers learn together using incident activities and local persistent
statistics. Predictions use fixed sums of layer activities. Training uses no
derivatives, backpropagation, propagated error vectors, or fitted classifier.

Official epoch-100 test accuracy (%), shown as mean ± sample SD and range over
four matched seeds for each update.

| Dataset | Learned layers | Semi-implicit Lion | Explicit Lion |
|---|---|---:|---:|
| MNIST | 3 fully connected layers, 2,560 neurons each | 98.14 ± 0.04 (98.10–98.19) | 98.19 ± 0.09 (98.06–98.27) |
| CIFAR-10 | 3 convolutional layers, 1,280 channels each | 80.69 ± 0.28 (80.41–81.08) | 80.67 ± 0.16 (80.45–80.82) |

Both variants train from scratch without dropout, image augmentation, or
weight averaging. The two updates use the same initialization and shuffled
image order within each seed. [results/paper.json](results/paper.json) links to
the 16 original test records and lists their final checkpoint hashes. The
[earlier single-seed dropout runs](results/historical-paper.json) and their
trajectories, selection records, and layer scores are archived in [results/](results/).
The official test sets have been reused.

## Installation

The original execution used Linux, Python 3.14, NumPy 2.5.3, Mojo 1.2.0, and an
NVIDIA RTX 4090. A compatible NVIDIA GPU and driver are required for the Mojo
backend. Portability to other GPU vendors has not been established.

```bash
uv sync --extra gpu
uv run --extra gpu python -m hebbian_networks.train mnist --dry-run
uv run --extra gpu python -m hebbian_networks.train cifar10 --dry-run
```

`uv.lock` pins the environment. `uv sync` installs the Python code and test
dependencies without the GPU toolchain. An existing Mojo 1.2.0 installation may
be selected with `MOJO_BIN=/path/to/mojo`; otherwise `mojo` is found on `PATH`.
Kernels compile on first use with floating-point contraction disabled. Build
outputs go into `.cache/build/`, configurable with `HEBBIAN_BUILD_DIR`.

## Reproduce the paper runs

Each command trains one dataset, seed, and Lion update. The default update is
semi-implicit. The default seeds are 10102 for MNIST and 10101 for CIFAR-10.

```bash
uv run --extra gpu python -m hebbian_networks.train mnist
uv run --extra gpu python -m hebbian_networks.train cifar10 --integration explicit
```

The commands download and checksum official data, recompute preprocessing
statistics from the complete training set, initialize fresh weights, and train
for the fixed 100 epochs. Run them sequentially. The default output path is
`runs/<dataset>/<integration>/seed<seed>/`. Each writes an atomic
epoch-boundary `resume.npz`, a final checkpoint, a trajectory, source/build
provenance, and an official-test result. Repeating the same command resumes the
existing run. SIGINT/SIGTERM stop after the current epoch has been saved.
`--output` and `--data-dir` accept explicit directories.

Both recipes use batch 32 and an initial step size of 0.00075 with the original
20-epoch plateau and 80-epoch cosine schedule. The schedule is indexed by
completed epochs, so epoch 21 also uses 0.00075. Epoch 100 uses a positive step
and the saved post-training step is zero. MNIST uses 128 winners per group and
margin 0.10. CIFAR-10 uses eight winners per group per location and margin 0.05.
Both recipes set dropout to zero. [MNIST recipe](configs/mnist.json) · [CIFAR-10 recipe](configs/cifar10.json)

Run the full matched comparison with both updates and all four seeds per dataset.

```bash
for integration in semi_implicit explicit; do
  for seed in 10102 10103 10104 10105; do
    uv run --extra gpu python -m hebbian_networks.train mnist --integration "$integration" --seed "$seed" || exit 1
  done
  for seed in 10101 10103 10104 10105; do
    uv run --extra gpu python -m hebbian_networks.train cifar10 --integration "$integration" --seed "$seed" || exit 1
  done
done
```

Official-test data are evaluated only after the requested full-data run finishes
epoch 100. New runs may differ across hardware and toolchains.

## Evaluate a final checkpoint

```bash
uv run --extra gpu python -m hebbian_networks.evaluate mnist runs/mnist/semi_implicit/seed10102/final.npz --layers
uv run --extra gpu python -m hebbian_networks.evaluate cifar10 runs/cifar10/explicit/seed10101/final.npz --layers
```

These commands also accept the original paper checkpoint format. `--layers`
reports individual-layer and cumulative scores using the fixed paper
normalization. All layers still compute their usual features; only the activities
included in the final score change. No weights are fitted or updated.

## Implementation and verification

- `hebbian_networks/mnist_connected_groups.py`: mixed fully connected network,
  checkpoint format, independent NumPy reference, and Mojo interface.
- `hebbian_networks/competitive_conv_connected_groups.py`: mixed convolutional
  model, independent reference, and original dense GPU implementation.
- `hebbian_networks/competitive_conv_connected_sparse.py`: the sparse accumulation
  backend used for both CIFAR-10 updates in the paper.
- `hebbian_networks/kernels/`: the original Mojo training kernels and their
  dependencies, including the GPU view and buffer helpers in `competitive_gru.mojo`.
- `common.py`, `data.py`, `mnist_reference.py`, `conv_reference.py`, `metrics.py`:
  the shared helpers extracted from the original experiment modules.
- `train.py`, `evaluate.py`, `scoring.py`: focused standalone entry points.
- [provenance/source-manifest.json](provenance/source-manifest.json): original
  source hashes and the scope of import/path/orchestration adaptations.

```bash
uv run pytest
uv run --extra gpu python -m hebbian_networks.verify --gpu
```

The first command checks numerical behavior and serialization on small synthetic
inputs. The second compiles the copied kernels and checks both dropout-free Lion
updates, forward activities, inference immutability, and checkpoint round trips
against independent NumPy calculations. It also checks compatibility with the
earlier dropout configurations.

The repository ships code and recorded results. The data files are downloaded
on demand and checkpoints are written by the training command. Source
`runs/...` paths in the result records identify the original experiment files.

Test and checkpoint verification are recorded in [provenance/verification.json](provenance/verification.json).

## License

[MIT](LICENSE).
