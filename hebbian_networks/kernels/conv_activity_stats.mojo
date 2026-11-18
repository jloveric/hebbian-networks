"""Class scores, nonzero activity statistics, and filter utilization."""
from std.math import fma, isfinite
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation
from gpu_buffer_utils import F, R, I


def column_scores_for_columns[columns: Int = 10](posts: R, energy: F, health: F, batch_arg: Int64,
                  locations_arg: Int64, width_arg: Int64, start_arg: Int64,
                  depth_arg: Int64, layer_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert posts.flat_rank == energy.flat_rank == health.flat_rank == 1
    var path, lane = block_idx.x, thread_idx.x
    var batch, width, locations = Int(batch_arg),Int(width_arg),Int(locations_arg)
    var column, image = divmod(path,batch)
    var sum = Float32(0)
    for coordinate in range(lane,locations*width,256):
        var g = posts[path*locations*width+coordinate]
        sum = fma(g,g,sum)
    var sums = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[256]())
    sums[lane] = sum
    barrier()
    comptime for shift in range(7,-1,-1):
        if lane < (1 << shift):
            sums[lane] += sums[lane+(1 << shift)]
        barrier()
    if lane == 0:
        energy[((Int(start_arg)+image)*columns+column)*Int(depth_arg)+Int(layer_arg)] = sums[0]
        if not isfinite(sums[0]):
            health[0] = 0


def observations_for_columns[columns: Int = 10](posts: R, nq: F, width_arg: Int64, locations_arg: Int64,
                 batch_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert posts.flat_rank == nq.flat_rank == 1
    var at = global_idx.x
    var width, locations, batch = Int(width_arg),Int(locations_arg),Int(batch_arg)
    if at < columns*batch*width:
        var path, j = divmod(at,width)
        var count, q = Float32(0),Float32(0)
        for location in range(locations):
            var g = posts[(path*locations+location)*width+j]
            count += Float32(1 if g != 0 else 0)
            q = fma(g,g,q)
        nq[2*at],nq[2*at+1] = count,q


def utilization_for_columns[columns: Int = 10](nq: R, seen: F, positive_seen: F, labels: I,
                batch_arg: Int64, width_arg: Int64, start_arg: Int64, train_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert nq.flat_rank == seen.flat_rank == positive_seen.flat_rank == labels.flat_rank == 1
    var at = global_idx.x
    var batch, width = Int(batch_arg),Int(width_arg)
    if at < columns*width:
        var column, j = divmod(at,width)
        var sum, positives = Float32(0),Float32(0)
        for image in range(batch):
            var count = nq[2*((column*batch+image)*width+j)]
            sum += count
            if train_arg != 0 and Int(labels[Int(start_arg)+image]) == column:
                positives += count
        seen[at] += sum
        positive_seen[at] += positives


comptime column_scores = column_scores_for_columns[10]
comptime observations = observations_for_columns[10]
comptime utilization = utilization_for_columns[10]
