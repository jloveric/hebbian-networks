"""Accumulate only nonzero selected observations in their original order."""
from std.math import ceildiv, fma
from max.gpu import block_idx, thread_idx
from gpu_buffer_utils import F, R, I, J
from conv_bias_patches import reflect_coordinate


def compact_selected_observations(posts: R, orientations: R, nq: R, scales: R,
        pairs: I, pair_counts: I, indices: J, values: F, counts: J,
        width_arg: Int64, locations_arg: Int64, batch_arg: Int64,
        start_arg: Int64, splits_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert posts.flat_rank == orientations.flat_rank == nq.flat_rank == scales.flat_rank == 1
    comptime assert pairs.flat_rank == pair_counts.flat_rank == indices.flat_rank == values.flat_rank == counts.flat_rank == 1
    var width,locations,batch,splits = Int(width_arg),Int(locations_arg),Int(batch_arg),Int(splits_arg)
    var j = block_idx.x*128+thread_idx.x
    if j >= width:
        return
    var group,part = divmod(block_idx.y,splits)
    var observations = Int(pair_counts[group])*locations
    var length = ceildiv(observations,splits)
    var capacity = ceildiv(batch*locations,splits)
    var key = (group*splits+part)*width+j
    var count = 0
    for observation in range(part*length,min(observations,(part+1)*length)):
        var pair,location = divmod(observation,locations)
        var image = Int(pairs[group*batch+pair])
        var row = image*locations+location
        var at = (group*batch*locations+row)*width+j
        var h = posts[at]
        if h != 0:
            var g = h*scales[(Int(start_arg)+image)*10+group]
            var n = nq[2*((group*batch+image)*width+j)]
            g /= n if n > 0 else 1
            var encoded = -row-1 if orientations[at] != 0 else row
            indices[key*capacity+count] = Int64(encoded)
            values[key*capacity+count] = g
            count += 1
    counts[key] = Int64(count)


def sparse_coactivity(inputs: R, indices: I, values: R, counts: I, controls: R,
        partials: F, incoming_arg: Int64, width_arg: Int64, locations_arg: Int64,
        batch_arg: Int64, kernel_arg: Int64, splits_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert inputs.flat_rank == indices.flat_rank == values.flat_rank == counts.flat_rank == controls.flat_rank == partials.flat_rank == 1
    var incoming,width,splits = Int(incoming_arg),Int(width_arg),Int(splits_arg)
    var first_i,j = block_idx.x*1024+thread_idx.x,block_idx.y
    if first_i >= incoming:
        return
    var group,part = divmod(block_idx.z,splits)
    if controls[2*(group*width+j)] == 0:
        return
    var capacity = ceildiv(Int(batch_arg)*Int(locations_arg),splits)
    var key = (group*splits+part)*width+j
    var total = SIMD[.float32,8](0)
    for slot in range(Int(counts[key])):
        var encoded = Int(indices[key*capacity+slot])
        var row = encoded if encoded >= 0 else -encoded-1
        var g = values[key*capacity+slot]
        comptime for r in range(8):
            var i = first_i+128*r
            if i < incoming:
                var coordinate = i if encoded >= 0 else reflect_coordinate(i,incoming,Int(kernel_arg))
                total[r] = fma(inputs[row*incoming+coordinate],g,total[r])
    comptime for r in range(8):
        var i = first_i+128*r
        if i < incoming:
            partials[((group*splits+part)*incoming+i)*width+j] = total[r]
