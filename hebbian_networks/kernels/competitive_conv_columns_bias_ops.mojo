"""Bias is an ordinary local constant-input synapse, never a class coordinate."""
from std.math import sqrt,fma
from max.gpu import block_idx,thread_idx,global_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major,stack_allocation
from competitive_gru import F,R,I


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


def bias_coactivity[shared_input: Bool](inputs: R, posts: R, orientations: R,
                 labels: I, scales: R, raw: F, incoming_arg: Int64, width_arg: Int64,
                 locations_arg: Int64, batch_arg: Int64, start_arg: Int64,
                 first_arg: Int64, size_arg: Int64, kernel_arg: Int64):
    # Each grid plane is one (column,image). This is direct u*g accumulation,
    # including the selected orientation's incoming synapse; no backward op.
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert inputs.flat_rank == posts.flat_rank == orientations.flat_rank == labels.flat_rank == scales.flat_rank == raw.flat_rank == 1
    var incoming, width, locations = Int(incoming_arg),Int(width_arg),Int(locations_arg)
    var batch, kernel = Int(batch_arg),Int(kernel_arg)
    var path, x, y = block_idx.z,thread_idx.x,thread_idx.y
    var column, chunk_image = divmod(path,Int(size_arg))
    var image = Int(first_arg)+chunk_image
    var post_path = column*batch+image
    var first_i, first_j = block_idx.y*32+y,block_idx.x*32+x
    var ordinary_u = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[32,16]())
    var reflected_u = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[32,16]())
    var ordinary_g = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[16,48]())
    var reflected_g = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[16,48]())
    var total = SIMD[DType.float32,4](0)
    var input_path = image if shared_input else post_path
    for start in range(0,locations,16):
        comptime for r in range(2):
            var i, j = first_i+16*r, first_j+16*r
            var a, b, g0, g1 = Float32(0),Float32(0),Float32(0),Float32(0)
            if i < incoming and start+x < locations:
                a = inputs[(input_path*locations+start+x)*incoming+i]
                b = inputs[(input_path*locations+start+x)*incoming+reflect_coordinate(i,incoming,kernel)]
            if j < width and start+y < locations:
                var at = (post_path*locations+start+y)*width+j
                var g = posts[at]
                if orientations[at] != 0:
                    g1 = g
                else:
                    g0 = g
            ordinary_u[y+16*r,x], reflected_u[y+16*r,x] = a,b
            ordinary_g[y,x+16*r], reflected_g[y,x+16*r] = g0,g1
        barrier()
        comptime for k in range(16):
            comptime for r in range(2):
                comptime for c in range(2):
                    total[r*2+c] = fma(ordinary_u[y+16*r,k],ordinary_g[k,x+16*c],total[r*2+c])
                    total[r*2+c] = fma(reflected_u[y+16*r,k],reflected_g[k,x+16*c],total[r*2+c])
        barrier()
    var sign = Float32(1 if Int(labels[Int(start_arg)+image]) == column else -1) if scales[(Int(start_arg)+image)*10+column] != 0 else Float32(0)
    comptime for r in range(2):
        comptime for c in range(2):
            var i,j = first_i+16*r,first_j+16*c
            if i < incoming and j < width:
                raw[(path*incoming+i)*width+j] = sign*total[r*2+c]


def mask_bias_features(inputs: F, mask: R, incoming_arg: Int64, channels_arg: Int64,
                         rows_arg: Int64, batch_arg: Int64, locations_arg: Int64, start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert inputs.flat_rank == mask.flat_rank == 1
    var at=global_idx.x
    var incoming,channels=Int(incoming_arg),Int(channels_arg)
    if at < Int(rows_arg)*incoming:
        var row,coordinate=divmod(at,incoming)
        var image=(row//Int(locations_arg))%Int(batch_arg)
        if coordinate < incoming-1:
            inputs[at]*=mask[(Int(start_arg)+image)*channels+coordinate%channels]
