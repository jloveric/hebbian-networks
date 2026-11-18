"""Connected class groups; saved forwards, pooled statistics, one Lion commit."""

from std.math import ceildiv, sqrt
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major, stack_allocation
from mnist_gpu_ops import F, R, J, Layer, view, clock_view, upload, download, project, select, route


def prepare_connected[first: Bool](source: R, target: F, start_arg: Int64,
                                   dimension_arg: Int64, previous_width_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == target.flat_rank == 3
    var c, b, lane = block_idx.z, block_idx.x, thread_idx.x
    var dimension, previous_width = Int(dimension_arg), Int(previous_width_arg)
    var squares = stack_allocation[.float32, address_space=.SHARED](row_major[256]())
    var total: Float32 = 0
    for i in range(lane, dimension, 256):
        var value: Float32 = 1
        if i < dimension-1:
            comptime if first:
                value = source[0, Int(start_arg)+b, i]
            else:
                var source_group, source_neuron = divmod(i, previous_width)
                value = source[source_group, b, source_neuron]
        target[c, b, i] = value
        total += value*value
    comptime if first:
        squares[lane] = total
        barrier()
        comptime for shift in range(7, -1, -1):
            if lane < 1 << shift:
                squares[lane] += squares[lane+(1 << shift)]
            barrier()
        var norm = sqrt(squares[0])
        for i in range(lane, dimension, 256):
            target[c, b, i] = target[c, b, i]/norm if norm > 0 else 0


def statistics(u: R, h: R, signs: R, a: F, q: F, n: F,
               dimension_arg: Int64, size_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert u.flat_rank == h.flat_rank == signs.flat_rank == a.flat_rank == q.flat_rank == n.flat_rank == 3
    var c, j, lane = block_idx.z, block_idx.x, thread_idx.x
    var count: Float32 = 0
    var square: Float32 = 0
    for b in range(Int(size_arg)):
        var g = h[c, b, j]
        if g != 0 and signs[0, b, c] != 0:
            count += 1
            square += g*g
    if lane == 0:
        n[0, c, j] = count
        q[0, c, j] = square/max(count, 1)
    for i in range(lane, Int(dimension_arg), 256):
        var value: Float32 = 0
        for b in range(Int(size_arg)):
            var g = h[c, b, j]
            var sign = signs[0, b, c]
            if g != 0 and sign != 0:
                value += sign*u[c, b, i]*g
        a[c, j, i] = value/max(count, 1)


def commit_integration(w: F, m: F, clock: J, a: R, q: R, n: R,
           dimension_arg: Int64, rate: Float32, history_mix: Float32, current_mix: Float32, explicit_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert w.flat_rank == m.flat_rank == clock.flat_rank == a.flat_rank == q.flat_rank == n.flat_rank == 3
    var c, j, lane = block_idx.z, block_idx.x, thread_idx.x
    if n[0, c, j] == 0:
        return
    var step = rate/sqrt(Float32(dimension_arg))
    var denominator = 1+rate*q[0, c, j]
    for i in range(lane, Int(dimension_arg), 256):
        var old_m = m[c, j, i]
        var value = a[c, j, i]
        var signal = history_mix*old_m+current_mix*value
        var direction = Float32(Int(signal > 0)-Int(signal < 0))
        var old_w = w[c, j, i]
        if explicit_arg != 0:
            w[c, j, i] = old_w+step*direction-rate*q[0, c, j]*old_w
        else:
            w[c, j, i] = (old_w+step*direction)/denominator
        m[c, j, i] = 0.99*old_m+0.01*value
    if lane == 0:
        clock[0, c, j] += 1


def count_usage(h: R, usage: J, size_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 64
    comptime assert h.flat_rank == usage.flat_rank == 3
    var c, j = block_idx.z, block_idx.x*64+thread_idx.x
    if j >= Int(h.dim[2]()):
        return
    var count: Int64 = 0
    for b in range(Int(size_arg)):
        count += Int64(Int(h[c, b, j] != 0))
    usage[0, c, j] += count


struct Pending(Movable):
    var a: DeviceBuffer[.float32]
    var q: DeviceBuffer[.float32]
    var n: DeviceBuffer[.float32]
    var usage: DeviceBuffer[.int64]

    def __init__(out self, ctx: DeviceContext, width: Int, dimension: Int) raises:
        self.a = ctx.enqueue_create_buffer[.float32](10*width*dimension)
        self.q = ctx.enqueue_create_buffer[.float32](10*width)
        self.n = ctx.enqueue_create_buffer[.float32](10*width)
        self.usage = ctx.enqueue_create_buffer[.int64](10*width)
        self.usage.enqueue_fill(0)


def run(packet: PythonObject) raises -> PythonObject:
    var ctx = DeviceContext()
    var config = packet["config"]
    var depth = Int(py=len(packet["state"]))
    var count = Int(py=packet["images"].shape[0])
    var inputs = Int(py=packet["images"].shape[1])
    var batch = Int(py=config["batch"])
    var train = Bool(py=packet["train"])
    var images = upload[.float32](ctx, packet["images"])
    var labels = upload[.int64](ctx, packet["labels"])
    var energy = ctx.enqueue_create_buffer[.float32](count*10*depth)
    var signs = ctx.enqueue_create_buffer[.float32](batch*10)
    var events = ctx.enqueue_create_buffer[.float32](count*2)
    events.enqueue_fill(0)
    var layers = List[Layer]()
    var pending = List[Pending]()
    for l in range(depth):
        layers.append(Layer(ctx, packet["state"][l], batch))
        pending.append(Pending(ctx, layers[l].width, layers[l].dimension))
    for start in range(0, count, batch):
        var size = min(batch, count-start)
        for l in range(depth):
            var width, dimension = layers[l].width, layers[l].dimension
            var u = view(layers[l].u, 10, size, dimension)
            var h = view(layers[l].h, 10, size, width)
            if l == 0:
                ctx.enqueue_function[prepare_connected[True]](view(images, 1, count, inputs).as_imm(), u,
                    Int64(start), Int64(dimension), Int64(inputs), grid_dim=(size, 1, 10), block_dim=256)
            else:
                ctx.enqueue_function[prepare_connected[False]](
                    view(layers[l-1].h, 10, size, layers[l-1].width).as_imm(), u,
                    Int64(0), Int64(dimension), Int64(layers[l-1].width), grid_dim=(size, 1, 10), block_dim=256)
            ctx.enqueue_function[project](u.as_imm(), view(layers[l].w, 10, width, dimension).as_imm(),
                h, Int64(size), Int64(width), Int64(dimension),
                grid_dim=(ceildiv(width, 32), ceildiv(size, 32), 10), block_dim=(16, 16))
            if width <= 256:
                ctx.enqueue_function[select[256]](h, view(energy, count, 10, depth), Int64(width),
                    Int64(py=config["winners"][l]), Int64(start), Int64(l), grid_dim=(size, 1, 10), block_dim=256)
            else:
                ctx.enqueue_function[select[512]](h, view(energy, count, 10, depth), Int64(width),
                    Int64(py=config["winners"][l]), Int64(start), Int64(l), grid_dim=(size, 1, 10), block_dim=512)
            ctx.enqueue_function[count_usage](h.as_imm(), clock_view(pending[l].usage, 10, width), Int64(size),
                grid_dim=(ceildiv(width, 64), 1, 10), block_dim=64)
        if train:
            var k = SIMD[.int64, 4](1)
            for l in range(depth):
                k[l] = Int64(py=config["winners"][l])
            ctx.enqueue_function[route](view(energy, count, 10, depth).as_imm(),
                TileTensor(labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), row_major((count,))).as_imm(),
                view(signs, 1, size, 10), view(events, 1, count, 2), Int64(start), Int64(size), Int64(depth),
                k[0], k[1], k[2], k[3], Float32(py=config["margin"]), Int64(0),
                grid_dim=ceildiv(size, 128), block_dim=128)
            for l in range(depth):
                var width, dimension = layers[l].width, layers[l].dimension
                ctx.enqueue_function[statistics](view(layers[l].u, 10, size, dimension).as_imm(),
                    view(layers[l].h, 10, size, width).as_imm(), view(signs, 1, size, 10).as_imm(),
                    view(pending[l].a, 10, width, dimension), view(pending[l].q, 1, 10, width),
                    view(pending[l].n, 1, 10, width), Int64(dimension), Int64(size),
                    grid_dim=(width, 1, 10), block_dim=256)
            for ll in range(depth):
                var l = depth-1-ll if Bool(py=packet["reverse"]) else ll
                var width, dimension = layers[l].width, layers[l].dimension
                ctx.enqueue_function[commit_integration](view(layers[l].w, 10, width, dimension),
                    view(layers[l].m, 10, width, dimension), clock_view(layers[l].clock, 10, width),
                    view(pending[l].a, 10, width, dimension).as_imm(), view(pending[l].q, 1, 10, width).as_imm(),
                    view(pending[l].n, 1, 10, width).as_imm(), Int64(dimension), Float32(py=packet["rate"]),
                    Float32(py=config["direction_mix"][0]), Float32(py=config["direction_mix"][1]),
                    Int64(1 if String(config["integration"]) == "explicit" else 0),
                    grid_dim=(width, 1, 10), block_dim=256)
    download[.float32](energy, packet["energies"])
    download[.float32](events, packet["events"])
    for l in range(depth):
        download[.int64](pending[l].usage, packet["usage"][l])
        if Int(py=len(packet["observations"])) > 0:
            download[.float32](layers[l].u, packet["observations"][l]["u"])
            download[.float32](layers[l].h, packet["observations"][l]["h"])
        if train:
            download[.float32](layers[l].w, packet["state"][l]["w"])
            download[.float32](layers[l].m, packet["state"][l]["m"])
            download[.int64](layers[l].clock, packet["state"][l]["clock"])
    ctx.synchronize()
    return packet


@export
def PyInit_mnist_connected_groups_native() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("mnist_connected_groups_native")
        module.def_function[run]("run")
        return module.finalize()
    except error:
        abort(String("Connected MNIST import: ", error))
