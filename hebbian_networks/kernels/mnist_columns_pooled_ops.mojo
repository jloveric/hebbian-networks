"""Neuron-owned minibatch statistics, as in the historical MNIST updater."""

from std.math import sqrt
from max.gpu import block_idx, thread_idx
from layout import MutTileTensor, ImmTileTensor, row_major

comptime L3 = type_of(row_major((Int(1), Int(1), Int(1))))
comptime F = MutTileTensor[.float32, L3, MutUntrackedOrigin]
comptime R = ImmTileTensor[.float32, L3, ImmUntrackedOrigin]
comptime J = MutTileTensor[.int64, L3, MutUntrackedOrigin]


def update_pooled(w: F, m: F, clock: J, u: R, h: R, scales: R,
                  dimension_arg: Int64, size_arg: Int64, rate: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert w.flat_rank == m.flat_rank == clock.flat_rank == u.flat_rank == h.flat_rank == scales.flat_rank == 3
    var c, j, lane = block_idx.z, block_idx.x, thread_idx.x
    var dimension, size = Int(dimension_arg), Int(size_arg)
    var count: Float32 = 0
    var q: Float32 = 0
    for b in range(size):
        var g = h[c, b, j]
        if g != 0 and scales[0, b, c] != 0:
            count += 1
            q += g*g
    if count == 0:
        return
    q /= count
    var step = rate/sqrt(Float32(dimension))
    var denominator = 1+rate*q
    for i in range(lane, dimension, 256):
        var a: Float32 = 0
        for b in range(size):
            var g = h[c, b, j]
            var sign = scales[0, b, c]
            if g != 0 and sign != 0:
                a += sign*u[c, b, i]*g
        a /= count
        var old_m = m[c, j, i]
        var signal = 0.125*old_m+0.875*a
        var direction = Float32(Int(signal > 0)-Int(signal < 0))
        w[c, j, i] = (w[c, j, i]+step*direction)/denominator
        m[c, j, i] = 0.99*old_m+0.01*a
    if lane == 0:
        clock[0, c, j] += 1
