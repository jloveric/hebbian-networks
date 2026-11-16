"""Exact FP32 forward alternatives: share projection inputs and fuse selection."""
from std.math import abs, fma, isfinite, min
from std.memory import bitcast
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from competitive_gru import F, R


def projection_orientations[shared_input: Bool, tile_k: Int = 16](inputs: R, weights: R, mirror: R,
        output: F, orientations: F, health: F,
        rows_arg: Int64, width_arg: Int64, inner_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert inputs.flat_rank == weights.flat_rank == mirror.flat_rank == output.flat_rank == orientations.flat_rank == health.flat_rank == 1
    comptime assert tile_k in (16,32)
    var rows, width, inner = Int(rows_arg), Int(width_arg), Int(inner_arg)
    var group, x, y = block_idx.z, thread_idx.x, thread_idx.y
    var first_row, first_j = block_idx.y*64+y, block_idx.x*64+x
    var left = stack_allocation[.float32, address_space=.SHARED](row_major[64,tile_k]())
    var right = stack_allocation[.float32, address_space=.SHARED](row_major[tile_k,80]())
    var reflected = stack_allocation[.float32, address_space=.SHARED](row_major[tile_k,80]())
    var ordinary_total = SIMD[.float32,16](0)
    var mirror_total = SIMD[.float32,16](0)
    var input_offset = 0 if shared_input else group*rows*inner
    for start in range(0,inner,tile_k):
        comptime for half in range(tile_k//16):
            comptime for r in range(4):
                var row, j = first_row+16*r, first_j+16*r
                var a, b, c = Float32(0), Float32(0), Float32(0)
                var kk = half*16
                if row < rows and start+kk+x < inner:
                    a = inputs[input_offset+row*inner+start+kk+x]
                if j < width and start+kk+y < inner:
                    var at = (group*inner+start+kk+y)*width+j
                    b, c = weights[at], mirror[at]
                left[y+16*r,kk+x], right[kk+y,x+16*r], reflected[kk+y,x+16*r] = a,b,c
        barrier()
        comptime for k in range(tile_k):
            comptime for r in range(4):
                comptime for c in range(4):
                    ordinary_total[r*4+c] = fma(left[y+16*r,k],right[k,x+16*c],ordinary_total[r*4+c])
                    mirror_total[r*4+c] = fma(left[y+16*r,k],reflected[k,x+16*c],mirror_total[r*4+c])
        barrier()
    comptime for r in range(4):
        comptime for c in range(4):
            var row, j = first_row+16*r,first_j+16*c
            if row < rows and j < width:
                var a, b = ordinary_total[r*4+c], mirror_total[r*4+c]
                var use_mirror = abs(b)>abs(a) or (abs(b)==abs(a) and b>a)
                var at = (group*rows+row)*width+j
                output[at] = b if use_mirror else a
                orientations[at] = 1 if use_mirror else 0
                if not isfinite(a) or not isfinite(b):
                    health[0] = 0


def select_packed[combine_orientations: Bool](activity: F, mirror: R, orientation: F,
        width_arg: Int64, winners_arg: Int64, padded_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 512
    comptime assert activity.flat_rank == mirror.flat_rank == orientation.flat_rank == 1
    var width, winners, padded = Int(width_arg),Int(winners_arg),Int(padded_arg)
    var row, lane = block_idx.x, thread_idx.x
    var value: Float32 = activity[row*width+lane] if lane < width else 0
    comptime if combine_orientations:
        if lane < width:
            var other = mirror[row*width+lane]
            var reflected = abs(other)>abs(value) or (abs(other)==abs(value) and other>value)
            orientation[row*width+lane] = 1 if reflected else 0
            if reflected:
                value = other
    var original_key = (UInt64(bitcast[.uint32](abs(value))) << 32) | UInt64(0xFFFFFFFF-lane) if lane < width else UInt64(0)
    var key = original_key
    var size = 2
    while size <= min(padded,WARP_SIZE):
        var stride = size//2
        while stride > 0:
            var other = warp.shuffle_xor(key,UInt32(stride))
            var want = ((lane & size) == 0) == ((lane & stride) == 0)
            if (other > key) == want:
                key = other
            stride //= 2
        size *= 2
    if padded >= WARP_SIZE and (lane & WARP_SIZE) != 0:
        key = warp.shuffle_xor(key,UInt32(WARP_SIZE-1))
    var keys = stack_allocation[.uint64, address_space=.SHARED](row_major[512]())
    keys[lane] = key
    barrier()
    var half = min(padded,WARP_SIZE)
    while half < padded:
        var local = lane%(2*half)
        var rank = local%half
        var keep = min(winners,half)
        var group = lane-local
        var other_start = group+half if local < half else group
        var own = keys[lane]
        var low, high = 0,keep
        if rank < keep:
            while low < high:
                var middle = (low+high)//2
                if keys[other_start+middle] > own:
                    low = middle+1
                else:
                    high = middle
        var destination = group+rank+low
        barrier()
        if rank < keep and rank+low < winners:
            keys[destination] = own
        barrier()
        half *= 2
    var threshold = keys[winners-1]
    if lane < width:
        activity[row*width+lane] = value if original_key >= threshold else 0


def masked_patches(source: R, output: F, mask: R, input_side_arg: Int64,
        side_arg: Int64, channels_arg: Int64, kernel_arg: Int64,
        batch_arg: Int64, start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == output.flat_rank == mask.flat_rank == 1
    var row, lane = block_idx.x, thread_idx.x
    var input_side, side = Int(input_side_arg),Int(side_arg)
    var cin, kernel = Int(channels_arg),Int(kernel_arg)
    var feature_dimension = cin*kernel*kernel
    var path, location = divmod(row,side*side)
    var y,x = divmod(location,side)
    var image = Int(start_arg)+path%Int(batch_arg)
    for coordinate in range(lane,feature_dimension+1,256):
        var value: Float32 = 1
        if coordinate < feature_dimension:
            var spatial,channel = divmod(coordinate,cin)
            var dy,dx = divmod(spatial,kernel)
            value = source[((path*input_side+y+dy)*input_side+x+dx)*cin+channel]*mask[image*cin+channel]
        output[row*(feature_dimension+1)+coordinate] = value
