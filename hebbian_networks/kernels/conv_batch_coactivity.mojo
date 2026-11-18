"""Direct minibatch coactivity and one local Lion update per active neuron."""
from std.math import ceildiv, fma
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from gpu_buffer_utils import F, R, I
from conv_bias_patches import reflect_coordinate


def batch_observations_for_columns[columns: Int = 10](nq: R, scales: R, controls: F, width_arg: Int64,
                       batch_arg: Int64, start_arg: Int64, image_balance_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var image_balance = image_balance_arg != 0
    var j, column, image = block_idx.x, block_idx.y, thread_idx.x
    var width, batch = Int(width_arg), Int(batch_arg)
    var n, q = Float32(0),Float32(0)
    if image < batch and scales[(Int(start_arg)+image)*columns+column] != 0:
        var at = 2*((column*batch+image)*width+j)
        n,q = nq[at],nq[at+1]
        if image_balance and n > 0:
            q /= n
            n = 1
    n,q = warp.sum(n),warp.sum(q)
    var sums = stack_allocation[.float32, address_space=.SHARED](row_major[2,4]())
    if image%WARP_SIZE == 0:
        sums[0,image//WARP_SIZE],sums[1,image//WARP_SIZE] = n,q
    barrier()
    if image < WARP_SIZE:
        n = sums[0,image] if image < 4 else 0
        q = sums[1,image] if image < 4 else 0
        n,q = warp.sum(n),warp.sum(q)
        if image == 0:
            var at = 2*(column*width+j)
            controls[at],controls[at+1] = n,q


def batch_coactivity[shared_input: Bool, columns: Int = 10](inputs: R, posts: R, orientations: R, observation_stats: R,
        scales: R, pairs: I, pair_counts: I, partials: F,
        incoming_arg: Int64, width_arg: Int64, locations_arg: Int64,
        batch_arg: Int64, start_arg: Int64, kernel_arg: Int64, splits_arg: Int64, image_balance_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    var image_balance = image_balance_arg != 0
    var incoming,width,locations = Int(incoming_arg),Int(width_arg),Int(locations_arg)
    var batch,kernel,splits = Int(batch_arg),Int(kernel_arg),Int(splits_arg)
    var column,part = divmod(block_idx.z,splits)
    var observations = Int(pair_counts[column])*locations
    if observations == 0:
        return
    var length = ceildiv(observations,splits)
    var first,last = part*length,min(observations,(part+1)*length)
    var x,y = thread_idx.x,thread_idx.y
    var first_i,first_j = block_idx.y*32+y,block_idx.x*32+x
    var ordinary_u = stack_allocation[.float32, address_space=.SHARED](row_major[32,16]())
    var reflected_u = stack_allocation[.float32, address_space=.SHARED](row_major[32,16]())
    var ordinary_g = stack_allocation[.float32, address_space=.SHARED](row_major[16,48]())
    var reflected_g = stack_allocation[.float32, address_space=.SHARED](row_major[16,48]())
    var total = SIMD[.float32,4](0)
    for start in range(first,last,16):
        comptime for r in range(2):
            var i,j = first_i+16*r,first_j+16*r
            var a,b,g0,g1 = Float32(0),Float32(0),Float32(0),Float32(0)
            if i < incoming and start+x < last:
                var pair,location = divmod(start+x,locations)
                var image = Int(pairs[column*batch+pair])
                var path = image if shared_input else column*batch+image
                a = inputs[(path*locations+location)*incoming+i]
                b = inputs[(path*locations+location)*incoming+reflect_coordinate(i,incoming,kernel)]
            if j < width and start+y < last:
                var pair,location = divmod(start+y,locations)
                var image = Int(pairs[column*batch+pair])
                var at = ((column*batch+image)*locations+location)*width+j
                var g = posts[at]*scales[(Int(start_arg)+image)*columns+column]
                if image_balance:
                    var n = observation_stats[2*((column*batch+image)*width+j)]
                    g /= n if n > 0 else 1
                if orientations[at] != 0:
                    g1 = g
                else:
                    g0 = g
            ordinary_u[y+16*r,x],reflected_u[y+16*r,x] = a,b
            ordinary_g[y,x+16*r],reflected_g[y,x+16*r] = g0,g1
        barrier()
        comptime for k in range(16):
            comptime for r in range(2):
                comptime for c in range(2):
                    total[r*2+c] = fma(ordinary_u[y+16*r,k],ordinary_g[k,x+16*c],total[r*2+c])
                    total[r*2+c] = fma(reflected_u[y+16*r,k],reflected_g[k,x+16*c],total[r*2+c])
        barrier()
    comptime for r in range(2):
        comptime for c in range(2):
            var i,j = first_i+16*r,first_j+16*c
            if i < incoming and j < width:
                partials[((column*splits+part)*incoming+i)*width+j] = total[r*2+c]




# Keep the original non-parametric kernel entry point for ten-class readers.
comptime batch_observations = batch_observations_for_columns[10]
