# Class-Guided Hebbian Networks

Standalone code for the jointly trained MNIST MLP and CIFAR-10 convolutional
network reported in the TMLR draft. Every neuron/filter is assigned a class;
all layers learn together using incident activities and local persistent
statistics. Predictions use fixed sums of layer activities. Training uses no
derivatives, backpropagation, propagated error vectors, or fitted classifier.

| Dataset | Learned layers | Paper seed | Official test accuracy |
|---|---|---:|---:|
| MNIST | 3 fully connected layers, 2,560 neurons each | 10102 | 98.34% |
| CIFAR-10 | 3 convolutional layers, 1,280 channels each | 10101 | 80.50% |

These are recorded results from the original runs, not newly trained results
from this extraction. [results/paper.json](results/paper.json) records the final
checkpoint hashes and validation-based reporting selection. Original trajectories,
selection records, and individual-layer scores are in [results/](results/).
Historical validation and official-test reuse applies. In particular, the original
MNIST runner trained and tested both full-data seeds; the paper reports the one
corresponding to the better preceding validation result.

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

## Reproduce the final paper runs

```bash
uv run --extra gpu python -m hebbian_networks.train mnist
uv run --extra gpu python -m hebbian_networks.train cifar10
```

The commands download and checksum official data, recompute preprocessing
statistics from the complete training set, initialize fresh weights, and train
for the fixed 100 epochs. Run them sequentially. Default output directories are
`runs/mnist/seed10102/` and `runs/cifar10/seed10101/`. Each writes an atomic
epoch-boundary `resume.npz`, a final checkpoint, a trajectory, source/build
provenance, and an official-test result. Repeating the same command resumes the
existing run. SIGINT/SIGTERM stop after the current epoch has been saved.
`--output` and `--data-dir` accept explicit directories.

The fixed recipes are [configs/mnist.json](configs/mnist.json) and
[configs/cifar10.json](configs/cifar10.json). Both use batch 32 and initial step
size 0.00075: constant through the first 20 epochs, then the original 80-epoch
cosine schedule. The schedule is indexed by completed epochs: epoch 21 also uses
0.00075, epoch 100 uses a positive step, and the saved post-training step is zero.
MNIST uses 30% hidden-input dropout, 128 winners per group, and margin 0.10.
CIFAR-10 uses 20% hidden-channel dropout, eight winners per group/location, and
margin 0.05. Neither uses image augmentation or weight averaging.

The standalone runner reproduces the selected final recipe directly; it does
not repeat the historical architecture and hyperparameter searches. To repeat
the other historical full-data MNIST seed explicitly:

```bash
uv run --extra gpu python -m hebbian_networks.train mnist --seed 10101
```

Official-test data are evaluated only after the requested full-data run finishes
epoch 100. The standalone command trains one requested seed; the original MNIST
experiment waited for both seeds before testing either. Its learning equations,
initialization, image orders, dropout streams, and final epoch are preserved.
New runs may differ across hardware/toolchains; the archived scores are not a
guarantee of bitwise reproduction in a different environment.

## Evaluate a final checkpoint

```bash
uv run --extra gpu python -m hebbian_networks.evaluate mnist runs/mnist/seed10102/final.npz --layers
uv run --extra gpu python -m hebbian_networks.evaluate cifar10 runs/cifar10/seed10101/final.npz --layers
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
  backend used for the final CIFAR-10 run, with unchanged checkpoint semantics.
- `hebbian_networks/kernels/`: 16 required Mojo sources copied verbatim, plus the extracted
  GPU view/buffer helpers in `competitive_gru.mojo`. That historical filename
  is retained for import compatibility; the recurrent model code is omitted.
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
inputs. The second compiles the copied kernels and checks forward activities,
local updates, deterministic dropout, inference immutability, and checkpoint
round trips against independent NumPy calculations. It uses bounded synthetic
minibatches and does not launch a paper training run.

Datasets, trained checkpoints, compiled binaries, historical sweeps, language
models, and manuscript sources are not included. Relative `runs/...` paths in
archived result JSON are provenance references, not required runtime inputs.

The extraction verification record is [provenance/verification.json](provenance/verification.json).
It records seven passing tests, the GPU checks, original-checkpoint compatibility,
and isolated wheel installation. Full dataset training was not rerun during extraction.

## License

[MIT](LICENSE).
