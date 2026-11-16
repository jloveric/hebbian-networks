"""Independent NumPy pooling and local convolutional update equations."""
import numpy as np

def reflected_indices(dimension,kernel):
    channels=(dimension-1)//(kernel*kernel)
    return np.concatenate((np.arange(dimension-1).reshape(kernel,kernel,channels)[:,::-1].reshape(-1),[dimension-1]))


def pool(h, mode="antialias"):
    """Explicit stride-one signed max, reflection padding, then fixed blur/stride2."""
    if mode != "antialias" or h.ndim != 5 or h.shape[2] != h.shape[3] or h.shape[2] < 2:
        raise ValueError("invalid signed blur-pool input")
    side = h.shape[2]
    dense = np.zeros((*h.shape[:2], side - 1, side - 1, h.shape[-1]), h.dtype)
    for dy in range(2):
        for dx in range(2):
            value = h[:, :, dy : dy + side - 1, dx : dx + side - 1, :]
            use = (np.abs(value) > np.abs(dense)) | (
                (np.abs(value) == np.abs(dense)) & (value > dense)
            )
            dense = np.where(use, value, dense)
    # NumPy's reflect degenerates to repeating the only element for size1.
    padded = np.pad(dense, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    small = side // 2
    result = np.zeros((*h.shape[:2], small, small, h.shape[-1]), h.dtype)
    taps = (1, 2, 1)
    for dy in range(3):
        for dx in range(3):
            result += padded[
                :, :, dy : dy + 2 * small : 2, dx : dx + 2 * small : 2, :
            ] * np.float32(taps[dy] * taps[dx] / 16)
    return result


def reference_update(state, records, labels, c, reverse=False, scales=None):
    """Explicit independent pair/location accumulation; no production reducers."""
    if scales is None:
        from .competitive_conv_connected_groups import selected_scores, training_scales
        energy = np.stack(
            [
                np.sum(r["h"].astype(np.float64) ** 2, axis=(2, 3, 4)).T.astype(np.float32)
                for r in records
            ],
            axis=-1,
        )
        side = records[0]["h"].shape[2] + c.kernels[0] - 1
        scales = training_scales(selected_scores(energy, c, side), labels, c)
    pending = []
    for bank, record, kernel in zip(state, records, c.kernels):
        incoming, width = bank["w"].shape[1:]
        numerator = np.zeros_like(bank["w"], dtype=np.float64)
        counts = np.zeros((c.columns, width), np.float64)
        squares = np.zeros_like(counts)
        reflected = reflected_indices(incoming, kernel)
        for column in range(c.columns):
            for image in range(len(labels)):
                sign = float(scales[image, column])
                if sign == 0:
                    continue
                inputs = record["u"][0 if record["u"].shape[0] == 1 else column, image].reshape(
                    -1, incoming
                )
                activity = record["h"][column, image].reshape(-1, width)
                orientation = record["orientation"][column, image].reshape(-1, width)
                image_raw = np.zeros((incoming, width), np.float64)
                image_n = np.zeros(width, np.float64)
                image_q = np.zeros(width, np.float64)
                for location in range(len(activity)):
                    g = activity[location].astype(np.float64)
                    u = inputs[location].astype(np.float64)
                    original = np.where(orientation[location] == 0, g, 0)
                    mirror = np.where(orientation[location] != 0, g, 0)
                    image_raw += sign * (np.outer(u, original) + np.outer(u[reflected], mirror))
                    image_n += g != 0
                    image_q += g * g
                if c.observation_balance == "image":
                    divisor = np.maximum(1, image_n)
                    numerator[column] += image_raw / divisor[None, :]
                    counts[column] += image_n > 0
                    squares[column] += image_q / divisor
                else:
                    numerator[column] += image_raw
                    counts[column] += image_n
                    squares[column] += image_q
        active = counts > 0
        divisor = np.maximum(1, counts)
        a = numerator / divisor[:, None, :]
        q = squares / divisor
        signal = c.direction_history * bank["m"].astype(np.float64) + (1 - c.direction_history) * a
        w = (bank["w"].astype(np.float64) + c.rate * np.sign(signal) / np.sqrt(incoming)) / (
            1 + c.rate * q[:, None, :]
        )
        m = c.history_decay * bank["m"].astype(np.float64) + (1 - c.history_decay) * a
        pending.append((active, w, m, numerator, counts, squares))
    for i in reversed(range(len(state))) if reverse else range(len(state)):
        active, w, m, _, _, _ = pending[i]
        state[i]["w"][...] = np.where(active[:, None, :], w, state[i]["w"]).astype(np.float32)
        state[i]["m"][...] = np.where(active[:, None, :], m, state[i]["m"]).astype(np.float32)
        state[i]["clock"] += active.astype(np.int64)
    return pending
