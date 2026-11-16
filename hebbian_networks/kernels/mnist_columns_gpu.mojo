"""Mojo 1.2 GPU forward snapshots and explicitly local dense-column updates."""

from std.math import abs, ceildiv, sqrt
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, MutTileTensor, ImmTileTensor, row_major, stack_allocation
from mnist_columns_pooled_ops import update_pooled
from mnist_columns_wide_select import select_wide

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


def keep_input(image_id: UInt32, coordinate: UInt32, seed: UInt32,
               epoch: UInt32, layer: UInt32, probability: Float32) -> Bool:
    var key = image_id ^ (seed*0x9E3779B9) ^ (epoch*0x85EBCA6B)
    key ^= (layer*0xC2B2AE35) ^ (coordinate*0x27D4EB2D)
    key = (key ^ (key >> 16))*0x7FEB352D
    key = (key ^ (key >> 15))*0x846CA68B
    key ^= key >> 16
    return Float32(key >> 8)/16777216 >= probability


def drop_competitors(fields: F, image_ids: I, start_arg: Int64, size_arg: Int64,
                     width_arg: Int64, seed: UInt32, epoch: UInt32, layer: UInt32,
                     probability: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert fields.flat_rank == 3 and image_ids.flat_rank == 1
    var at = global_idx.x
    var size, width = Int(size_arg), Int(width_arg)
    if at >= 10*size*width:
        return
    var row, neuron = divmod(at, width)
    var column, b = divmod(row, size)
    var keep = keep_input(UInt32(image_ids[Int(start_arg)+b]), UInt32(neuron), seed, epoch, layer, probability)
    fields[column, b, neuron] = fields[column, b, neuron]/(1-probability) if keep else 0


def prepare[first: Bool](source: R, target: F, start_arg: Int64, size_arg: Int64,
                        dimension_arg: Int64, unit_transfer: Int64, bias: Float32,
                        image_ids: I, global_start: Int64, seed: UInt32, epoch: UInt32,
                        layer: Int64, probability: Float32, image_dropout: Int64, training: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == target.flat_rank == 3
    comptime assert image_ids.flat_rank == 1
    var start = Int(start_arg)
    var size = Int(size_arg)
    var dimension = Int(dimension_arg)
    var c, b, lane = block_idx.z, block_idx.x, thread_idx.x
    var squares = stack_allocation[.float32, address_space=.SHARED](row_major[256]())
    var total: Float32 = 0
    for i in range(lane, dimension, 256):
        var value = bias
        if i < dimension-1:
            comptime if first:
                value = source[0, start+b, i]
            else:
                value = source[c, b, i]
        target[c, b, i] = value
        if i < dimension-1:
            total += value*value
    comptime if first:
        if lane == 0:
            total += bias*bias
    if first or unit_transfer != 0:
        squares[lane] = total
        barrier()
        comptime for shift in range(7, -1, -1):
            if lane < 1 << shift:
                squares[lane] += squares[lane+(1 << shift)]
            barrier()
        var norm = sqrt(squares[0])
        for i in range(lane, dimension, 256):
            comptime if first:
                target[c, b, i] = target[c, b, i]/norm if norm > 0 else 0
            else:
                if i < dimension-1:
                    target[c, b, i] = target[c, b, i]/norm if norm > 0 else 0
    if training != 0 and probability > 0 and (not first or image_dropout != 0):
        var image_id = UInt32(image_ids[Int(global_start)+b])
        for i in range(lane, dimension-1, 256):
            target[c, b, i] = target[c, b, i]/(1-probability) if keep_input(
                image_id, UInt32(i), seed, epoch, UInt32(layer), probability) else 0


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


def update(w: F, m: F, clock: J, u: R, h: R, scales: R,
           width_arg: Int64, dimension_arg: Int64, size_arg: Int64, rate: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert w.flat_rank == m.flat_rank == clock.flat_rank == u.flat_rank == h.flat_rank == scales.flat_rank == 3
    var width = Int(width_arg)
    var dimension = Int(dimension_arg)
    var size = Int(size_arg)
    var c, j, lane = block_idx.z, block_idx.x, thread_idx.x
    var step = rate/sqrt(Float32(dimension))
    for b in range(size):
        var g = h[c, b, j]
        var sign = scales[0, b, c]
        if g == 0 or sign == 0:
            continue
        var denominator = 1+rate*g*g
        for i in range(lane, dimension, 256):
            var a = sign*u[c, b, i]*g
            var old_m = m[c, j, i]
            var signal = 0.125*old_m+0.875*a
            var direction = Float32(Int(signal > 0)-Int(signal < 0))
            w[c, j, i] = (w[c, j, i]+step*direction)/denominator
            m[c, j, i] = 0.99*old_m+0.01*a
        if lane == 0:
            clock[0, c, j] += 1


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


def run(packet: PythonObject) raises -> PythonObject:
    var ctx = DeviceContext()
    var config = packet["config"]
    var depth = Int(py=len(packet["state"]))
    var count = Int(py=packet["images"].shape[0])
    var inputs = Int(py=packet["images"].shape[1])
    var batch = Int(py=config["batch"])
    var train = Bool(py=packet["train"])
    var unit_transfer = Int64(Int(String(py=config["transfer"]) == "unit"))
    var bias = Float32(py=config["bias_scale"])
    var images = upload[.float32](ctx, packet["images"])
    var labels = upload[.int64](ctx, packet["labels"])
    var image_ids = upload[.int64](ctx, packet["image_ids"])
    var seed, epoch = UInt32(py=packet["seed"]), UInt32(py=packet["epoch"])
    var probability = Float32(py=config["dropout"])
    var competition_dropout = String(py=config.get("dropout_placement", PythonObject("incoming"))) == "competition"
    var input_probability: Float32 = 0 if competition_dropout else probability
    var image_dropout = Int64(Int(Bool(py=config["dropout_input"])))
    var energy = ctx.enqueue_create_buffer[.float32](count*10*depth)
    var scales = ctx.enqueue_create_buffer[.float32](batch*10)
    var events = ctx.enqueue_create_buffer[.float32](count*2)
    events.enqueue_fill(0)
    var layers = List[Layer]()
    for l in range(depth):
        layers.append(Layer(ctx, packet["state"][l], batch))
    for start in range(0, count, batch):
        var size = min(batch, count-start)
        for l in range(depth):
            var width, dimension = layers[l].width, layers[l].dimension
            var layer_bias = Float32(py=config["first_bias_scale"]) if l == 0 else bias
            # Form the device view at each launch so the owner stays live while enqueuing.
            var id_view = TileTensor(image_ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), row_major((count,))).as_imm()
            var u = view(layers[l].u, 10, size, dimension)
            var h = view(layers[l].h, 10, size, width)
            if l == 0:
                ctx.enqueue_function[prepare[True]](view(images, 1, count, inputs).as_imm(), u,
                    Int64(start), Int64(size), Int64(dimension), unit_transfer, layer_bias,
                    id_view, Int64(start), seed, epoch, Int64(l), input_probability, image_dropout, Int64(Int(train)),
                    grid_dim=(size, 1, 10), block_dim=256)
            else:
                ctx.enqueue_function[prepare[False]](
                    view(layers[l-1].h, 10, size, layers[l-1].width).as_imm(), u,
                    Int64(0), Int64(size), Int64(dimension), unit_transfer, layer_bias,
                    id_view, Int64(start), seed, epoch, Int64(l), input_probability, image_dropout, Int64(Int(train)),
                    grid_dim=(size, 1, 10), block_dim=256)
            ctx.enqueue_function[project](u.as_imm(), view(layers[l].w, 10, width, dimension).as_imm(),
                h, Int64(size), Int64(width), Int64(dimension), grid_dim=(ceildiv(width, 32), ceildiv(size, 32), 10), block_dim=(16, 16))
            if train and competition_dropout and probability > 0:
                ctx.enqueue_function[drop_competitors](h, id_view, Int64(start), Int64(size),
                    Int64(width), seed, epoch, UInt32(l), probability,
                    grid_dim=ceildiv(10*size*width, 128), block_dim=128)
            var k = Int(py=config["winners"][l])
            if width <= 32:
                ctx.enqueue_function[select[32]](h, view(energy, count, 10, depth), Int64(width), Int64(k), Int64(start), Int64(l),
                    grid_dim=(size, 1, 10), block_dim=32)
            elif width <= 256:
                ctx.enqueue_function[select[256]](h, view(energy, count, 10, depth), Int64(width), Int64(k), Int64(start), Int64(l),
                    grid_dim=(size, 1, 10), block_dim=256)
            elif width <= 512:
                ctx.enqueue_function[select[512]](h, view(energy, count, 10, depth), Int64(width), Int64(k), Int64(start), Int64(l),
                    grid_dim=(size, 1, 10), block_dim=512)
            elif width <= 1024:
                ctx.enqueue_function[select[1024]](h, view(energy, count, 10, depth), Int64(width), Int64(k), Int64(start), Int64(l),
                    grid_dim=(size, 1, 10), block_dim=1024)
            else:
                ctx.enqueue_function[select_wide](h, view(energy, count, 10, depth), Int64(width), Int64(k), Int64(start), Int64(l),
                    grid_dim=(size, 1, 10), block_dim=256)
        if train:
            var k = SIMD[.int64, 4](1)
            for l in range(depth):
                k[l] = Int64(py=config["winners"][l])
            ctx.enqueue_function[route](view(energy, count, 10, depth).as_imm(),
                TileTensor(labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), row_major((count,))).as_imm(), view(scales, 1, size, 10),
                view(events, 1, count, 2), Int64(start), Int64(size), Int64(depth), k[0], k[1], k[2], k[3],
                Float32(py=config["margin"]), Int64(Int(String(py=config["score"]) == "final")),
                grid_dim=ceildiv(size, 128), block_dim=128)
            for ll in range(depth):
                var l = depth-1-ll if Bool(py=packet["reverse"]) else ll
                var width, dimension = layers[l].width, layers[l].dimension
                if String(py=config["aggregation"]) == "pooled":
                    ctx.enqueue_function[update_pooled](view(layers[l].w, 10, width, dimension),
                        view(layers[l].m, 10, width, dimension), clock_view(layers[l].clock, 10, width),
                        view(layers[l].u, 10, size, dimension).as_imm(),
                        view(layers[l].h, 10, size, width).as_imm(), view(scales, 1, size, 10).as_imm(),
                        Int64(dimension), Int64(size), Float32(py=packet["rate"]),
                        grid_dim=(width, 1, 10), block_dim=256)
                else:
                    ctx.enqueue_function[update](view(layers[l].w, 10, width, dimension),
                        view(layers[l].m, 10, width, dimension), clock_view(layers[l].clock, 10, width),
                        view(layers[l].u, 10, size, dimension).as_imm(),
                        view(layers[l].h, 10, size, width).as_imm(), view(scales, 1, size, 10).as_imm(),
                        Int64(width), Int64(dimension), Int64(size), Float32(py=packet["rate"]),
                        grid_dim=(width, 1, 10), block_dim=256)
    download[.float32](energy, packet["energies"])
    download[.float32](events, packet["events"])
    if train:
        for l in range(depth):
            download[.float32](layers[l].w, packet["state"][l]["w"])
            download[.float32](layers[l].m, packet["state"][l]["m"])
            download[.int64](layers[l].clock, packet["state"][l]["clock"])
    ctx.synchronize()
    return packet


@export
def PyInit_mnist_columns_gpu_native() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("mnist_columns_gpu_native")
        module.def_function[run]("run")
        return module.finalize()
    except e:
        abort(String("MNIST columns GPU import: ", e))
