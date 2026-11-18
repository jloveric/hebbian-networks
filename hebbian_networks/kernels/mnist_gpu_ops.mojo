"""GPU projection, competition, routing, and buffer helpers for MNIST."""

from std.math import abs
from std.python import PythonObject
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, MutTileTensor, ImmTileTensor, row_major, stack_allocation


comptime L3 = type_of(row_major((Int(1), Int(1), Int(1))))
comptime L1 = type_of(row_major((Int(1),)))
comptime F = MutTileTensor[.float32, L3, MutUntrackedOrigin]
comptime R = ImmTileTensor[.float32, L3, ImmUntrackedOrigin]
comptime I = ImmTileTensor[.int64, L1, ImmUntrackedOrigin]
comptime J = MutTileTensor[.int64, L3, MutUntrackedOrigin]


def view(mut b: DeviceBuffer[.float32], a: Int, n: Int, d: Int) -> F:
    return TileTensor(b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), row_major((a, n, d)))


def clock_view(mut b: DeviceBuffer[.int64], n: Int, d: Int) -> J:
    return TileTensor(b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), row_major((Int(1), n, d)))


def upload[dtype: DType](ctx: DeviceContext, a: PythonObject) raises -> DeviceBuffer[dtype]:
    var n = Int(py=a.size)
    var p = ImmPointer[Scalar[dtype], ImmUntrackedOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    var host = ctx.enqueue_create_host_buffer[dtype](n)
    for i in range(n):
        host[i] = p[unsafe_offset=i]
    var result = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(result, host)
    return result^


def download[dtype: DType](buf: DeviceBuffer[dtype], a: PythonObject) raises:
    var n = Int(py=a.size)
    if n > len(buf):
        raise Error("host destination exceeds device buffer")
    var p = MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    with buf.map_to_host() as host:
        for i in range(n):
            p[unsafe_offset=i] = host[i]


def project(u: R, w: R, output: F, size_arg: Int64, width_arg: Int64, dimension_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert u.flat_rank == w.flat_rank == output.flat_rank == 3
    var size = Int(size_arg)
    var width = Int(width_arg)
    var dimension = Int(dimension_arg)
    var c, x, y = block_idx.z, thread_idx.x, thread_idx.y
    var b, j = block_idx.y*32+y, block_idx.x*32+x
    var left = stack_allocation[.float32, address_space=.SHARED](row_major[32, 16]())
    var right = stack_allocation[.float32, address_space=.SHARED](row_major[32, 16]())
    var sums = SIMD[.float32, 4](0)
    for start in range(0, dimension, 16):
        comptime for r in range(2):
            var ub: Float32 = 0
            var wb: Float32 = 0
            if b+16*r < size and start+x < dimension:
                ub = u[c, b+16*r, start+x]
            if j+16*r < width and start+y < dimension:
                wb = w[c, j+16*r, start+y]
            left[y+16*r, x] = ub
            right[x+16*r, y] = wb
        barrier()
        comptime for k in range(16):
            comptime for r in range(2):
                comptime for t in range(2):
                    sums[r*2+t] += left[y+16*r, k]*right[x+16*t, k]
        barrier()
    comptime for r in range(2):
        comptime for t in range(2):
            if b+16*r < size and j+16*t < width:
                output[c, b+16*r, j+16*t] = sums[r*2+t]


def select[PAD: Int](h: F, energy: F, width_arg: Int64, winners_arg: Int64, start_arg: Int64, layer_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = PAD
    comptime assert h.flat_rank == energy.flat_rank == 3
    var width = Int(width_arg)
    var winners = Int(winners_arg)
    var start = Int(start_arg)
    var layer = Int(layer_arg)
    var c, b, lane = block_idx.z, block_idx.x, thread_idx.x
    var values = stack_allocation[.float32, address_space=.SHARED](row_major[PAD]())
    var ids = stack_allocation[.int32, address_space=.SHARED](row_major[PAD]())
    var sums = stack_allocation[.float32, address_space=.SHARED](row_major[PAD]())
    values[lane] = abs(h[c, b, lane]) if lane < width else -1
    ids[lane] = Int32(lane)
    barrier()
    var size = 2
    while size <= PAD:
        var stride = size//2
        while stride > 0:
            var other = lane ^ stride
            var v = values[other]
            var j = ids[other]
            var mine = values[lane]
            var myid = ids[lane]
            var better = v > mine or (v == mine and j < myid)
            var want = ((lane & size) == 0) == ((lane & stride) == 0)
            barrier()
            if better == want:
                values[lane], ids[lane] = v, j
            barrier()
            stride //= 2
        size *= 2
    if lane >= winners and lane < width:
        h[c, b, Int(ids[lane])] = 0
    barrier()
    var g: Float32 = 0
    if lane < width:
        g = h[c, b, lane]
    sums[lane] = g*g
    barrier()
    var stride = PAD//2
    while stride > 0:
        if lane < stride:
            sums[lane] += sums[lane+stride]
        barrier()
        stride //= 2
    if lane == 0:
        energy[start+b, c, layer] = sums[0]


def route(energy: R, labels: I, scales: F, events: F, start_arg: Int64, size_arg: Int64,
          depth_arg: Int64, k0_arg: Int64, k1_arg: Int64, k2_arg: Int64, k3_arg: Int64, margin: Float32, final_only_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert energy.flat_rank == scales.flat_rank == events.flat_rank == 3
    comptime assert labels.flat_rank == 1
    var start = Int(start_arg)
    var size = Int(size_arg)
    var depth = Int(depth_arg)
    var k0 = Int(k0_arg)
    var k1 = Int(k1_arg)
    var k2 = Int(k2_arg)
    var k3 = Int(k3_arg)
    var final_only = Int(final_only_arg)
    var b = global_idx.x
    if b >= size:
        return
    var y = Int(labels[start+b])
    var ss = SIMD[.float32, 16](0)
    var ks = SIMD[.int64, 4](Int64(k0), Int64(k1), Int64(k2), Int64(k3))
    for c in range(10):
        for l in range(depth):
            if final_only == 0 or l == depth-1:
                ss[c] += energy[start+b, c, l]/Float32(ks[l])
        scales[0, b, c] = 0
    var rival = 1 if y == 0 else 0
    var prediction = 0
    for c in range(10):
        if c != y and ss[c] > ss[rival]:
            rival = c
        if ss[c] > ss[prediction]:
            prediction = c
    var separation = (ss[y]-ss[rival])/(ss[y]+ss[rival]+1e-30)
    var learn = prediction != y or separation < margin
    events[0, start+b, 0] = Float32(Int(learn))
    events[0, start+b, 1] = Float32(Int(prediction != y))
    if learn:
        scales[0, b, y] = 1
        scales[0, b, rival] = -1


struct Layer(Movable):
    var w: DeviceBuffer[.float32]
    var m: DeviceBuffer[.float32]
    var clock: DeviceBuffer[.int64]
    var u: DeviceBuffer[.float32]
    var h: DeviceBuffer[.float32]
    var width: Int
    var dimension: Int

    def __init__(out self, ctx: DeviceContext, bank: PythonObject, batch: Int) raises:
        self.width = Int(py=bank["w"].shape[1])
        self.dimension = Int(py=bank["w"].shape[2])
        self.w = upload[.float32](ctx, bank["w"])
        self.m = upload[.float32](ctx, bank["m"])
        self.clock = upload[.int64](ctx, bank["clock"])
        self.u = ctx.enqueue_create_buffer[.float32](10*batch*self.dimension)
        self.h = ctx.enqueue_create_buffer[.float32](10*batch*self.width)
