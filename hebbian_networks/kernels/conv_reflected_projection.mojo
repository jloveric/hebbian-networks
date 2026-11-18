"""FP32 first-layer projection with shared inputs and reflected filters."""
from std.math import abs, fma, isfinite
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from gpu_buffer_utils import F, R


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
