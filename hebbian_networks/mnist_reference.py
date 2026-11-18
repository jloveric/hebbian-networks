"""Neuron-local minibatch statistics used by the MNIST reference model."""
import numpy as np

def statistics(u, h, signs):
    """Only incident input/output products are merged at each receiving neuron."""
    observed = (h != 0) & (signs[:, :, None] != 0)
    n = observed.sum(0)
    signed_h = h.astype(np.float64)*signs[:, :, None]
    # This contraction adds observations of the same synapse; it fits nothing.
    raw = np.einsum("bci,bcj->cji", u.astype(np.float64), signed_h)
    a = (raw/np.maximum(1, n[:, :, None])).astype(np.float32)
    q = ((h.astype(np.float64)**2*observed).sum(0)/np.maximum(1, n)).astype(np.float32)
    return a, q, n


def commit(bank, a, q, n, rate, *, direction_mix=(.125, .875), integration="semi_implicit"):
    active = n > 0
    w, m = bank["w"], bank["m"]
    d = np.sign(np.float32(direction_mix[0])*m+np.float32(direction_mix[1])*a)
    step = np.float32(rate)/np.float32(np.sqrt(w.shape[-1]))
    damping = np.float32(rate)*q[:, :, None]
    if integration == "explicit":
        updated = w+step*d-damping*w
    else:
        updated = (w+step*d)/(1+damping)
    bank["w"][active] = updated[active]
    bank["m"][active] = (np.float32(.99)*m+np.float32(.01)*a)[active]
    bank["clock"] += active
