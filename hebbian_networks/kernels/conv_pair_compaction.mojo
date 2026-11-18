"""Compact supervised image pairs for CIFAR-10 coactivity accumulation."""
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from gpu_buffer_utils import R, J


def compact_pairs_for_columns[columns: Int = 10](scales: R, pairs: J, counts: J, batch_arg: Int64, start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var column, lane = block_idx.x, thread_idx.x
    var batch = Int(batch_arg)
    var scan = stack_allocation[.int32, address_space=.SHARED](row_major[128]())
    var included = lane < batch and scales[(Int(start_arg)+lane)*columns+column] != 0
    scan[lane] = 1 if included else 0
    barrier()
    comptime for shift in range(7):
        var offset = 1 << shift
        var previous = Int32(0)
        if lane >= offset:
            previous = scan[lane-offset]
        barrier()
        scan[lane] += previous
        barrier()
    if included:
        pairs[column*batch+Int(scan[lane])-1] = Int64(lane)
    if lane == 127:
        counts[column] = Int64(scan[lane])


comptime compact_pairs = compact_pairs_for_columns[10]
