"""Signed top-k selection with several items per GPU thread."""

from std.math import abs
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from layout import MutTileTensor, row_major, stack_allocation

comptime L3 = type_of(row_major((Int(1), Int(1), Int(1))))
comptime F = MutTileTensor[.float32, L3, MutUntrackedOrigin]


def select_wide(h: F, energy: F, width_arg: Int64, winners_arg: Int64,
                start_arg: Int64, layer_arg: Int64):
    comptime PAD = 2048
    comptime THREADS = 256
    comptime ITEMS = PAD//THREADS
    comptime MAX_THREADS_PER_BLOCK_METADATA = THREADS
    comptime assert h.flat_rank == energy.flat_rank == 3
    var width, winners = Int(width_arg), Int(winners_arg)
    var start, layer = Int(start_arg), Int(layer_arg)
    var c, b, lane = block_idx.z, block_idx.x, thread_idx.x
    var values = stack_allocation[.float32, address_space=.SHARED](row_major[PAD]())
    var ids = stack_allocation[.int32, address_space=.SHARED](row_major[PAD]())
    var sums = stack_allocation[.float32, address_space=.SHARED](row_major[PAD]())
    comptime for item in range(ITEMS):
        var index = lane+item*THREADS
        values[index] = abs(h[c, b, index]) if index < width else -1
        ids[index] = Int32(index)
    barrier()
    var size = 2
    while size <= PAD:
        var stride = size//2
        while stride > 0:
            var replacements = SIMD[.bool, ITEMS](fill=False)
            var incoming_values = SIMD[.float32, ITEMS](0)
            var incoming_ids = SIMD[.int32, ITEMS](0)
            # Snapshot every partner before writing either half of a pair.
            comptime for item in range(ITEMS):
                var index = lane+item*THREADS
                var other = index ^ stride
                var value, identity = values[other], ids[other]
                var mine, my_id = values[index], ids[index]
                var better = value > mine or (value == mine and identity < my_id)
                var want = ((index & size) == 0) == ((index & stride) == 0)
                replacements[item] = better == want
                incoming_values[item], incoming_ids[item] = value, identity
            barrier()
            comptime for item in range(ITEMS):
                if replacements[item]:
                    var index = lane+item*THREADS
                    values[index], ids[index] = incoming_values[item], incoming_ids[item]
            barrier()
            stride //= 2
        size *= 2
    comptime for item in range(ITEMS):
        var index = lane+item*THREADS
        if index >= winners and index < width:
            h[c, b, Int(ids[index])] = 0
    barrier()
    comptime for item in range(ITEMS):
        var index = lane+item*THREADS
        var value: Float32 = 0
        if index < width:
            value = h[c, b, index]
        sums[index] = value*value
    barrier()
    var stride = PAD//2
    while stride > 0:
        comptime for item in range(ITEMS):
            var index = lane+item*THREADS
            if index < stride:
                sums[index] += sums[index+stride]
        barrier()
        stride //= 2
    if lane == 0:
        energy[start+b, c, layer] = sums[0]
