"""MNIST and CIFAR-10 only, with training-only preprocessing and source checksums."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from .common import REPO, preprocess

TRAIN_IMAGES, TEST_IMAGES = 60000, 10000
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=lambda a: a.tolist()) + "\n"
    )
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def download(url, md5, directory):
    path = directory / url.rsplit("/", 1)[1]
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Downloading " + url, flush=True)
        temporary = path.with_suffix(path.suffix + ".download")
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset and digest(temporary, "md5") == md5:
            temporary.replace(path)
        else:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-"} if offset else {}
            )
            with urllib.request.urlopen(request, timeout=120) as incoming:
                partial = incoming.status == 206
                if partial and not incoming.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("unexpected partial-download response")
                # Servers without Range support return 200: safely restart that file.
                with temporary.open("ab" if partial else "wb") as output:
                    while block := incoming.read(1024 * 1024):
                        output.write(block)
        if path.exists():
            return path
        if digest(temporary, "md5") != md5:
            temporary.unlink()
            raise ValueError("official dataset checksum mismatch: " + url)
        temporary.replace(path)
    if digest(path, "md5") != md5:
        raise ValueError("cached source checksum mismatch: " + str(path))
    return path


def read_pair(raw_source, *, test=False):
    files = (("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 16),
             ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 8)) if test else (
             ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 16),
             ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 8))
    arrays, sources = [], {}
    for name, expected, offset in files:
        raw = (Path(raw_source)/name).read_bytes()
        if hashlib.md5(raw).hexdigest() != expected:
            raise ValueError("MNIST source checksum mismatch")
        sources[name] = hashlib.sha256(raw).hexdigest()
        arrays.append(np.frombuffer(gzip.decompress(raw)[offset:], np.uint8).copy())
    images = arrays[0].reshape(-1, 784)/np.float32(255)
    labels = arrays[1]
    expected_size = TEST_IMAGES if test else TRAIN_IMAGES
    if len(images) != expected_size or labels.shape != (expected_size,):
        raise ValueError("Unexpected complete MNIST partition size")
    return images, labels, sources


def load_training(raw_source):
    images, labels, sources = read_pair(raw_source)
    mean = images.mean(0, dtype=np.float64).astype(np.float32)
    return dict(images=preprocess(images, mean), labels=labels, mean=mean, sources=sources)


def load_test(raw_source, mean):
    if mean.shape != (784,) or mean.dtype != np.float32 or not np.isfinite(mean).all():
        raise ValueError("Test requires the saved full-training mean")
    images, labels, sources = read_pair(raw_source, test=True)
    return dict(images=preprocess(images, mean), labels=labels, sources=sources)


def decode_test_batch(raw):
    if len(raw) != 10000*3073:
        raise ValueError("official test must contain exactly 10,000 images")
    rows = np.frombuffer(raw, np.uint8).reshape(10000, 3073)
    labels = rows[:,0].astype(np.int64)
    np.testing.assert_array_equal(np.bincount(labels, minlength=10), np.full(10,1000))
    return rows[:,1:].reshape(10000,3,32,32).transpose(0,2,3,1), labels


MNIST_SOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def prepare_mnist(directory, *, test=False):
    directory = Path(directory)
    for name, md5 in MNIST_SOURCES.items():
        if name.startswith("t10k") == test:
            download(MNIST_URL + name, md5, directory)
    return directory


def mnist_training(directory):
    return load_training(prepare_mnist(directory))


def mnist_test(directory, mean):
    return load_test(prepare_mnist(directory, test=True), mean)


def cifar_archive(directory):
    return download(CIFAR_URL, CIFAR_MD5, Path(directory))


def cifar_training(directory):
    archive = cifar_archive(directory)
    batches = []
    with tarfile.open(archive, "r:gz") as source:
        for i in range(1, 6):
            with source.extractfile(f"cifar-10-batches-bin/data_batch_{i}.bin") as member:
                batches.append(np.frombuffer(member.read(), np.uint8).reshape(10000, 3073))
    rows = np.concatenate(batches)
    labels = rows[:, 0].astype(np.int32)
    pixels = rows[:, 1:].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(np.bincount(labels, minlength=10), np.full(10, 5000))
    x = pixels.astype(np.float32) / np.float32(255)
    mean = x.mean((0, 1, 2), dtype=np.float64).astype(np.float32)
    x -= mean
    sample = np.concatenate([np.flatnonzero(labels == k)[:400] for k in range(10)])
    return dict(images=x, labels=labels, mean=mean, sample=sample,
                sources={archive.name: digest(archive)})


def cifar_test(directory, mean):
    if mean.shape != (3,) or mean.dtype != np.float32 or not np.isfinite(mean).all():
        raise ValueError("Test requires the saved full-training RGB mean")
    archive = cifar_archive(directory)
    with tarfile.open(archive, "r:gz") as source:
        with source.extractfile("cifar-10-batches-bin/test_batch.bin") as member:
            images, labels = decode_test_batch(member.read())
    return dict(images=images.astype(np.float32) / np.float32(255) - mean,
                labels=labels, sources={archive.name: digest(archive)})
