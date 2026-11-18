"""Bias is an ordinary local constant-input synapse, never a class coordinate."""
from std.math import sqrt, fma
from max.gpu import block_idx,thread_idx,global_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major,stack_allocation
from gpu_buffer_utils import F, R


def reflect_coordinate(row: Int, incoming: Int, kernel: Int) -> Int:
    if row == incoming-1:
        return row
    var channels = (incoming-1)//(kernel*kernel)
    var spatial, channel = divmod(row, channels)
    var y, x = divmod(spatial, kernel)
    return (y*kernel+kernel-1-x)*channels+channel


def reflect_bias_columns(w: R, mirror: F, incoming_arg: Int64, width_arg: Int64,
                    kernel_arg: Int64, columns_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert w.flat_rank == mirror.flat_rank == 1
    var incoming, width, kernel = Int(incoming_arg), Int(width_arg), Int(kernel_arg)
    var at = global_idx.x
    if at < Int(columns_arg)*incoming*width:
        var column, coordinate = divmod(at, incoming*width)
        var row, j = divmod(coordinate, width)
        mirror[at] = w[(column*incoming+reflect_coordinate(row,incoming,kernel))*width+j]


def bias_patches[first: Bool](source: R, output: F, input_side_arg: Int64,
                        side_arg: Int64, channels_arg: Int64, kernel_arg: Int64,
                        start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == output.flat_rank == 1
    var row, lane = block_idx.x, thread_idx.x
    var input_side, side = Int(input_side_arg), Int(side_arg)
    var cin, kernel = Int(channels_arg), Int(kernel_arg)
    var feature_dimension = cin*kernel*kernel
    var dimension = feature_dimension+1
    var path, location = divmod(row, side*side)
    var y, x = divmod(location, side)
    var source_path = path
    comptime if first:
        source_path += Int(start_arg)
    var sum = Float32(0)
    for coordinate in range(lane, dimension, 256):
        var value = Float32(1)
        if coordinate < feature_dimension:
            var spatial, channel = divmod(coordinate, cin)
            var dy, dx = divmod(spatial, kernel)
            value = source[((source_path*input_side+y+dy)*input_side+x+dx)*cin+channel]
            sum = fma(value,value,sum)
        output[row*dimension+coordinate] = value
    comptime if first:
        var squares = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[256]())
        squares[lane] = sum
        barrier()
        comptime for shift in range(7,-1,-1):
            if lane < (1 << shift):
                squares[lane] += squares[lane+(1 << shift)]
            barrier()
        var norm = sqrt(squares[0]+Float32(1))
        for coordinate in range(lane, dimension, 256):
            output[row*dimension+coordinate] = output[row*dimension+coordinate]/norm if norm > 0 else 0
