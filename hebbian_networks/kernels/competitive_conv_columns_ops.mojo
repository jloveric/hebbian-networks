"""Grouped class-column forward and direct local patch/activity coactivities."""
from std.math import sqrt, fma, isfinite
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation
from competitive_gru import F, R, I, J


def reflect_coordinate(row: Int, incoming: Int, kernel: Int) -> Int:
    var channels = incoming//(kernel*kernel)
    var spatial, channel = divmod(row, channels)
    var y, x = divmod(spatial, kernel)
    return (y*kernel+kernel-1-x)*channels+channel


def reflect_columns(w: R, mirror: F, incoming_arg: Int64, width_arg: Int64,
                    kernel_arg: Int64, columns_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert w.flat_rank == mirror.flat_rank == 1
    var incoming, width, kernel = Int(incoming_arg), Int(width_arg), Int(kernel_arg)
    var at = global_idx.x
    if at < Int(columns_arg)*incoming*width:
        var column, coordinate = divmod(at, incoming*width)
        var row, j = divmod(coordinate, width)
        mirror[at] = w[(column*incoming+reflect_coordinate(row,incoming,kernel))*width+j]


def patches[first: Bool](source: R, output: F, input_side_arg: Int64,
                        side_arg: Int64, channels_arg: Int64, kernel_arg: Int64,
                        start_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == output.flat_rank == 1
    var row, lane = block_idx.x, thread_idx.x
    var input_side, side = Int(input_side_arg), Int(side_arg)
    var cin, kernel = Int(channels_arg), Int(kernel_arg)
    var dimension = cin*kernel*kernel
    var path, location = divmod(row, side*side)
    var y, x = divmod(location, side)
    var source_path = path
    comptime if first:
        source_path += Int(start_arg)
    var sum = Float32(0)
    for coordinate in range(lane, dimension, 256):
        var spatial, channel = divmod(coordinate, cin)
        var dy, dx = divmod(spatial, kernel)
        var value = source[((source_path*input_side+y+dy)*input_side+x+dx)*cin+channel]
        output[row*dimension+coordinate] = value
        sum = fma(value,value,sum)
    comptime if first:
        var squares = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[256]())
        squares[lane] = sum
        barrier()
        comptime for shift in range(7,-1,-1):
            if lane < (1 << shift):
                squares[lane] += squares[lane+(1 << shift)]
            barrier()
        var norm = sqrt(squares[0])
        for coordinate in range(lane, dimension, 256):
            output[row*dimension+coordinate] = output[row*dimension+coordinate]/norm if norm > 0 else 0


def grouped_projection[shared_input: Bool](inputs: R, weights: R, output: F,
                    health: F, rows_arg: Int64, width_arg: Int64, inner_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert inputs.flat_rank == weights.flat_rank == output.flat_rank == health.flat_rank == 1
    var rows, width, inner = Int(rows_arg), Int(width_arg), Int(inner_arg)
    var group, x, y = block_idx.z, thread_idx.x, thread_idx.y
    var first_row, first_j = block_idx.y*64+y, block_idx.x*64+x
    var left = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[64,16]())
    var right = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[16,80]())
    var total = SIMD[DType.float32,16](0)
    var input_offset = 0 if shared_input else group*rows*inner
    for start in range(0,inner,16):
        comptime for r in range(4):
            var row, j = first_row+16*r, first_j+16*r
            var av, bv = Float32(0), Float32(0)
            if row < rows and start+x < inner:
                av = inputs[input_offset+row*inner+start+x]
            if j < width and start+y < inner:
                bv = weights[(group*inner+start+y)*width+j]
            left[y+16*r,x], right[y,x+16*r] = av,bv
        barrier()
        comptime for k in range(16):
            comptime for r in range(4):
                comptime for c in range(4):
                    total[r*4+c] = fma(left[y+16*r,k],right[k,x+16*c],total[r*4+c])
        barrier()
    comptime for r in range(4):
        comptime for c in range(4):
            var row, j = first_row+16*r,first_j+16*c
            if row < rows and j < width:
                output[(group*rows+row)*width+j] = total[r*4+c]
                if not isfinite(total[r*4+c]):
                    health[0] = 0


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


def local_coactivity[shared_input: Bool](inputs: R, posts: R, orientations: R,
                 labels: I, raw: F, incoming_arg: Int64, width_arg: Int64,
                 locations_arg: Int64, batch_arg: Int64, start_arg: Int64,
                 first_arg: Int64, size_arg: Int64, kernel_arg: Int64):
    # Each grid plane is one (column,image). This is direct u*g accumulation,
    # including the selected orientation's incoming synapse; no backward op.
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert inputs.flat_rank == posts.flat_rank == orientations.flat_rank == labels.flat_rank == raw.flat_rank == 1
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
    var sign = Float32(1 if Int(labels[Int(start_arg)+image]) == column else -1)
    comptime for r in range(2):
        comptime for c in range(2):
            var i,j = first_i+16*r,first_j+16*c
            if i < incoming and j < width:
                raw[(path*incoming+i)*width+j] = sign*total[r*2+c]


def compose_columns(pending_w: F, pending_m: F, events: F,
                    raw: R, nq: R, labels: I, health: F,
                    incoming_arg: Int64, width_arg: Int64, batch_arg: Int64,
                    start_arg: Int64, first_arg: Int64, size_arg: Int64,
                    rate: Float32, negative_scale: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert pending_w.flat_rank == pending_m.flat_rank == events.flat_rank == raw.flat_rank == nq.flat_rank == labels.flat_rank == health.flat_rank == 1
    var at = global_idx.x
    var incoming,width,batch = Int(incoming_arg),Int(width_arg),Int(batch_arg)
    if at >= 10*incoming*width:
        return
    var column, coordinate = divmod(at,incoming*width)
    var j = coordinate%width
    var weight, history = pending_w[at],pending_m[at]
    var steps = Float32(0)
    var scale = sqrt(Float32(incoming))
    for chunk_image in range(Int(size_arg)):
        var image = Int(first_arg)+chunk_image
        var count = nq[2*((column*batch+image)*width+j)]
        if count == 0:
            continue
        var a = raw[(column*Int(size_arg)+chunk_image)*incoming*width+coordinate]/count
        var q = nq[2*((column*batch+image)*width+j)+1]/count
        var eta = rate if Int(labels[Int(start_arg)+image]) == column else rate*negative_scale
        var signal = fma(Float32(.875),a,Float32(.125)*history)
        var direction = Float32(1 if signal > 0 else -1 if signal < 0 else 0)/scale
        weight = (weight+eta*direction)/(1+eta*q)
        history = fma(Float32(.01),a,Float32(.99)*history)
        steps += 1
        if not isfinite(weight) or not isfinite(history) or not isfinite(q):
            health[0] = 0
    pending_w[at],pending_m[at] = weight,history
    if coordinate < width:
        events[column*width+j] += steps


def commit_columns(w: F, m: F, clock: J, pending_w: R, pending_m: R,
                   events: R, incoming_arg: Int64, width_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert w.flat_rank == m.flat_rank == clock.flat_rank == pending_w.flat_rank == pending_m.flat_rank == events.flat_rank == 1
    var at = global_idx.x
    var incoming,width = Int(incoming_arg),Int(width_arg)
    if at < 10*incoming*width:
        var column, coordinate = divmod(at,incoming*width)
        if events[column*width+coordinate%width] > 0:
            w[at],m[at] = pending_w[at],pending_m[at]
    if at < 10*width:
        clock[at] += Int64(events[at])


# Keep the original non-parametric kernel entry point for ten-class readers.
comptime column_scores = column_scores_for_columns[10]


# Keep the original non-parametric kernel entry point for ten-class readers.
comptime observations = observations_for_columns[10]


# Keep the original non-parametric kernel entry point for ten-class readers.
comptime utilization = utilization_for_columns[10]
