"""Exact signed top-k for up to256 channels, one warp per spatial location."""
from std.math import abs, max, min
from std.memory import bitcast
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from competitive_gru import F,R


def select_warp[combine_orientations: Bool](activity: F, mirror: R, orientation: F,
        width_arg: Int64, winners_arg: Int64, rows_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var row = block_idx.x*4+thread_idx.x//WARP_SIZE
    var lane = thread_idx.x%WARP_SIZE
    if row >= Int(rows_arg):
        return
    var width,winners = Int(width_arg),Int(winners_arg)
    var values = SIMD[.float32,8](0)
    var original = SIMD[.uint64,8](0)
    var keys = SIMD[.uint64,8](0)
    comptime for r in range(8):
        var j = lane+WARP_SIZE*r
        if j < width:
            var value = activity[row*width+j]
            comptime if combine_orientations:
                var other = mirror[row*width+j]
                var reflected = abs(other)>abs(value) or (abs(other)==abs(value) and other>value)
                orientation[row*width+j] = 1 if reflected else 0
                if reflected:
                    value = other
            values[r] = value
            original[r] = (UInt64(bitcast[.uint32](abs(value))) << 32) | UInt64(0xFFFFFFFF-j)
            keys[r] = original[r]
    # Sort eight register values within each lane, then merge only the global
    # winners using warp reductions. No block-level barriers or shared sort.
    comptime for level in range(1,4):
        comptime for shift in range(level-1,-1,-1):
            comptime for r in range(8):
                comptime other = r^(1 << shift)
                comptime if r < other:
                    var a,b = keys[r],keys[other]
                    comptime if (r & (1 << level)) == 0:
                        keys[r],keys[other] = max(a,b),min(a,b)
                    else:
                        keys[r],keys[other] = min(a,b),max(a,b)
    var threshold = UInt64(0)
    for rank in range(winners):
        threshold = warp.max(keys[0])
        if keys[0] == threshold:
            comptime for r in range(7):
                keys[r] = keys[r+1]
            keys[7] = 0
    comptime for r in range(8):
        var j = lane+WARP_SIZE*r
        if j < width:
            activity[row*width+j] = values[r] if original[r] >= threshold else 0
