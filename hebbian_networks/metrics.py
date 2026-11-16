"""Ordinary deployed predictions; evaluation never updates model state."""
import numpy as np
from . import mnist_connected_groups as mnist
from . import competitive_conv_connected_sparse as cifar

def mnist_metrics(state, x, y, c):
    before = mnist.common.fingerprint(state)
    energy, events = mnist.run(state, x, c)
    assert before == mnist.common.fingerprint(state)
    score = mnist.scores(energy, c)
    prediction = score.argmax(1)
    other = score.copy()
    true = score[np.arange(len(y)), y]
    other[np.arange(len(y)), y] = -np.inf
    wrong = other.max(1)
    errors = int((prediction != y).sum())
    return dict(images=len(y), errors=errors, accuracy=1-errors/len(y),
                layer_accuracy=[float((energy[:, :, l].argmax(1) == y).mean()) for l in range(len(state))],
                per_class_accuracy=[float((prediction[y == k] == k).mean()) if np.any(y == k) else None
                                    for k in range(10)],
                mean_margin=float(((true-wrong)/(true+wrong+1e-30)).mean()),
                all_zero_fraction=float(np.all(score == 0, axis=1).mean()),
                numerical_health="finite",
                utilization=[dict(fraction_ever_selected=float((u > 0).mean()),
                                  never_selected=int((u == 0).sum()),
                                  selection_count_min=int(u.min()), selection_count_max=int(u.max()),
                                  selection_count_median=float(np.median(u))) for u in events["usage"]])


def cifar_metrics(state,images,labels,c):
    before=cifar.fingerprint(state)
    packet=cifar.score(state,images,c,evaluation_batch=128)
    assert before==cifar.fingerprint(state)
    energy=packet['energies'];scores=cifar.selected_scores(energy,c,images.shape[1]);prediction=scores.argmax(1)
    true=scores[np.arange(len(labels)),labels]
    other=scores.copy();other[np.arange(len(labels)),labels]=-np.inf
    wrong=other.max(1)
    return dict(images=len(labels),accuracy=float((prediction==labels).mean()),errors=int((prediction!=labels).sum()),
        per_class_accuracy=[float((prediction[labels==k]==k).mean()) if np.any(labels==k) else None for k in range(10)],
        layer_accuracy=[float((energy[:,:,l].argmax(1)==labels).mean()) for l in range(len(state))],
        mean_margin=float(((true-wrong)/(true+wrong+1e-30)).mean()),
        predicted_counts=np.bincount(prediction,minlength=10).tolist(),
        all_zero_fraction=float(np.all(scores==0,axis=1).mean()),
        utilization=[float((a>0).mean()) for a in packet['utilization']])


