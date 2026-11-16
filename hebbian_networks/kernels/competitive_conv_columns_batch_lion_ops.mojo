"""Direct minibatch coactivity and one local Lion update per active neuron."""
from std.math import abs, ceildiv, fma, isfinite, sqrt
from max.gpu import block_idx, global_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from competitive_gru import F, R, I, J
from competitive_conv_columns_bias_ops import reflect_coordinate


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


def batch_observations(nq: R, scales: R, controls: F, width_arg: Int64,
                       batch_arg: Int64, start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var j, column, image = block_idx.x, block_idx.y, thread_idx.x
    var width, batch = Int(width_arg), Int(batch_arg)
    var n, q = Float32(0),Float32(0)
    if image < batch and scales[(Int(start_arg)+image)*10+column] != 0:
        var at = 2*((column*batch+image)*width+j)
        n,q = nq[at],nq[at+1]
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


def batch_coactivity[shared_input: Bool](inputs: R, posts: R, orientations: R,
        scales: R, pairs: I, pair_counts: I, partials: F,
        incoming_arg: Int64, width_arg: Int64, locations_arg: Int64,
        batch_arg: Int64, start_arg: Int64, kernel_arg: Int64, splits_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
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
                var g = posts[at]*scales[(Int(start_arg)+image)*10+column]
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


def apply_batch_lion(weights: F, history: F, clocks: J, partials: R, controls: R,
        numerator: F, health: F, incoming_arg: Int64, width_arg: Int64,
        splits_arg: Int64, rate: Float32, trace_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var at = global_idx.x
    var incoming,width,splits = Int(incoming_arg),Int(width_arg),Int(splits_arg)
    if at >= 10*incoming*width:
        return
    var column,coordinate = divmod(at,incoming*width)
    var j = coordinate%width
    var n = controls[2*(column*width+j)]
    if n == 0:
        if trace_arg != 0:
            numerator[at] = 0
        return
    var total = Float32(0)
    for part in range(splits):
        total += partials[(column*splits+part)*incoming*width+coordinate]
    if trace_arg != 0:
        numerator[at] = total
    var a,q = total/n,controls[2*(column*width+j)+1]/n
    var old_m = history[at]
    var signal = fma(Float32(.875),a,Float32(.125)*old_m)
    var direction = Float32(1 if signal > 0 else -1 if signal < 0 else 0)/sqrt(Float32(incoming))
    var weight = (weights[at]+rate*direction)/(1+rate*q)
    var next_m = fma(Float32(.01),a,Float32(.99)*old_m)
    weights[at],history[at] = weight,next_m
    if coordinate < width:
        clocks[column*width+j] += 1
    if not isfinite(weight) or not isfinite(next_m) or not isfinite(q):
        health[0] = 0


# Keep the original non-parametric kernel entry point for ten-class readers.
comptime compact_pairs = compact_pairs_for_columns[10]
