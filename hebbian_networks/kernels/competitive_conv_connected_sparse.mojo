"""Connected class groups; shared patches and one pooled local commit."""
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul.vendor.blas import matmul
from competitive_gru import F, R, fv, rv, iv, jv, upload, download
from competitive_conv_columns_bias_ops import bias_patches, reflect_bias_columns
from competitive_conv_columns_performance_ops import projection_orientations
from competitive_conv_columns_warp_selection import select_warp
from competitive_conv_columns_antialias_ops import antialias_pool
from competitive_conv_columns_ops import column_scores, observations, utilization
from competitive_conv_columns_margin_ops import choose_margin_columns
from competitive_conv_columns_batch_lion_ops import compact_pairs
from competitive_conv_columns_batch_lion_image_ops import batch_observations, batch_coactivity
from competitive_conv_columns_batch_lion_history_ops import apply_batch_lion
from competitive_conv_connected_sparse_ops import compact_selected_observations, sparse_coactivity


def connected_patches(source: R, output: F, masks: R, input_side_arg: Int64,
        side_arg: Int64, previous_width_arg: Int64, kernel_arg: Int64,
        batch_arg: Int64, start_arg: Int64, masked_arg: Int64):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 256
    comptime assert source.flat_rank == output.flat_rank == masks.flat_rank == 1
    var row, lane = block_idx.x, thread_idx.x
    var side, input_side = Int(side_arg), Int(input_side_arg)
    var previous_width, kernel, batch = Int(previous_width_arg), Int(kernel_arg), Int(batch_arg)
    var channels = 10*previous_width
    var dimension = kernel*kernel*channels+1
    var image, location = divmod(row, side*side)
    var y, x = divmod(location, side)
    for coordinate in range(lane, dimension, 256):
        var value = Float32(1)
        if coordinate < dimension-1:
            var spatial, channel = divmod(coordinate, channels)
            var dy, dx = divmod(spatial, kernel)
            var group, neuron = divmod(channel, previous_width)
            value = source[(((group*batch+image)*input_side+y+dy)*input_side+x+dx)*previous_width+neuron]
            if masked_arg != 0:
                value *= masks[(Int(start_arg)+image)*channels+channel]
        output[row*dimension+coordinate] = value


struct Layer(Movable):
    var w: DeviceBuffer[.float32]
    var m: DeviceBuffer[.float32]
    var clock: DeviceBuffer[.int64]
    var mirror_w: DeviceBuffer[.float32]
    var inputs: DeviceBuffer[.float32]
    var posts: DeviceBuffer[.float32]
    var mirror_posts: DeviceBuffer[.float32]
    var orientation: DeviceBuffer[.float32]
    var pooled: DeviceBuffer[.float32]
    var selected_indices: DeviceBuffer[.int64]
    var selected_values: DeviceBuffer[.float32]
    var selected_counts: DeviceBuffer[.int64]
    var raw: DeviceBuffer[.float32]
    var controls: DeviceBuffer[.float32]
    var numerator: DeviceBuffer[.float32]
    var nq: DeviceBuffer[.float32]
    var seen: DeviceBuffer[.float32]
    var positive_seen: DeviceBuffer[.float32]
    var incoming: Int
    var width: Int
    var input_side: Int
    var side: Int

    def __init__(out self, ctx: DeviceContext, bank: PythonObject, input_side: Int,
            kernel: Int, batch: Int, train: Bool, splits: Int, trace: Bool) raises:
        self.incoming, self.width = Int(py=bank["w"].shape[1]), Int(py=bank["w"].shape[2])
        self.input_side, self.side = input_side, input_side-kernel+1
        self.w = upload[.float32](ctx, bank["w"])
        self.m = upload[.float32](ctx, bank["m"]) if train else ctx.enqueue_create_buffer[.float32](1)
        self.clock = upload[.int64](ctx, bank["clock"]) if train else ctx.enqueue_create_buffer[.int64](1)
        var weights = 10*self.incoming*self.width
        var rows = batch*self.side*self.side
        self.mirror_w = ctx.enqueue_create_buffer[.float32](weights)
        self.inputs = ctx.enqueue_create_buffer[.float32](rows*self.incoming)
        self.posts = ctx.enqueue_create_buffer[.float32](10*rows*self.width)
        self.mirror_posts = ctx.enqueue_create_buffer[.float32](10*rows*self.width)
        self.orientation = ctx.enqueue_create_buffer[.float32](10*rows*self.width)
        self.pooled = ctx.enqueue_create_buffer[.float32](max(1,10*batch*(self.side//2)*(self.side//2)*self.width))
        var selected_capacity = 10*splits*self.width*ceildiv(rows,splits) if train else 1
        self.selected_indices = ctx.enqueue_create_buffer[.int64](selected_capacity)
        self.selected_values = ctx.enqueue_create_buffer[.float32](selected_capacity)
        self.selected_counts = ctx.enqueue_create_buffer[.int64](10*splits*self.width if train else 1)
        self.raw = ctx.enqueue_create_buffer[.float32](splits*weights if train else 1)
        self.controls = ctx.enqueue_create_buffer[.float32](20*self.width if train else 1)
        self.numerator = ctx.enqueue_create_buffer[.float32](weights if trace else 1)
        self.nq = ctx.enqueue_create_buffer[.float32](20*batch*self.width)
        self.seen = ctx.enqueue_create_buffer[.float32](10*self.width)
        self.positive_seen = ctx.enqueue_create_buffer[.float32](10*self.width)
        self.seen.enqueue_fill(0)
        self.positive_seen.enqueue_fill(0)


def run(packet: PythonObject) raises -> PythonObject:
    var ctx = DeviceContext()
    var c = packet["config"]
    var train, trace = Bool(py=packet["train"]), Bool(py=packet["trace"])
    var count, depth = Int(py=packet["images"].shape[0]), Int(py=len(packet["state"]))
    var batch = Int(py=c["batch"])
    if not train:
        batch = Int(py=packet["evaluation_batch"])
    var images, labels = upload[.float32](ctx,packet["images"]), upload[.int64](ctx,packet["labels"])
    var scales = upload[.float32](ctx,packet["scales"])
    var weights = upload[.float32](ctx,packet["score_weights"])
    var pairs = ctx.enqueue_create_buffer[.int64](10*batch)
    var pair_counts = ctx.enqueue_create_buffer[.int64](10)
    var energy = ctx.enqueue_create_buffer[.float32](count*10*depth)
    var health = ctx.enqueue_create_buffer[.float32](1)
    health.enqueue_fill(1)
    var layers = List[Layer]()
    var masks = List[DeviceBuffer[.float32]]()
    var side = Int(py=packet["images"].shape[1])
    for i in range(depth):
        masks.append(upload[.float32](ctx,packet["masks"][i]))
        layers.append(Layer(ctx,packet["state"][i],side,Int(py=c["kernels"][i]),batch,
            train,Int(py=packet["splits"][i]),trace))
        side = layers[i].side
        if Bool(py=c["pool_after"][i]):
            side //= 2
    for start in range(0,count,batch):
        var size = min(batch,count-start)
        for i in range(depth):
            var kernel = Int(py=c["kernels"][i])
            var di, width, locations = layers[i].incoming, layers[i].width, layers[i].side*layers[i].side
            var rows = size*locations
            ctx.enqueue_function[reflect_bias_columns](rv(layers[i].w),fv(layers[i].mirror_w),
                Int64(di),Int64(width),Int64(kernel),Int64(10),grid_dim=ceildiv(10*di*width,128),block_dim=128)
            if i == 0:
                ctx.enqueue_function[bias_patches[True]](rv(images),fv(layers[i].inputs),
                    Int64(layers[i].input_side),Int64(layers[i].side),Int64((di-1)//(kernel*kernel)),
                    Int64(kernel),Int64(start),grid_dim=rows,block_dim=256)
                ctx.enqueue_function[projection_orientations[True,32]](rv(layers[i].inputs),rv(layers[i].w),
                    rv(layers[i].mirror_w),fv(layers[i].posts),fv(layers[i].orientation),fv(health),
                    Int64(rows),Int64(width),Int64(di),grid_dim=(ceildiv(width,64),ceildiv(rows,64),10),block_dim=(16,16))
            else:
                var source = rv(layers[i-1].pooled) if Bool(py=c["pool_after"][i-1]) else rv(layers[i-1].posts)
                ctx.enqueue_function[connected_patches](source,fv(layers[i].inputs),rv(masks[i]),
                    Int64(layers[i].input_side),Int64(layers[i].side),Int64(layers[i-1].width),
                    Int64(kernel),Int64(size),Int64(start),Int64(1 if train and Float32(py=c["dropout_probability"])>0 else 0),
                    grid_dim=rows,block_dim=256)
                var lhs = TileTensor(layers[i].inputs.unsafe_ptr(),row_major((rows,di))).as_imm()
                for group in range(10):
                    var rhs = TileTensor(layers[i].w.unsafe_ptr().unsafe_offset(group*di*width),row_major((di,width))).as_imm()
                    var target = TileTensor(layers[i].posts.unsafe_ptr().unsafe_offset(group*rows*width),row_major((rows,width)))
                    matmul[use_tf32=False](ctx,target,lhs,rhs,c_row_major=True)
                    var reflected_rhs = TileTensor(layers[i].mirror_w.unsafe_ptr().unsafe_offset(group*di*width),row_major((di,width))).as_imm()
                    var reflected_target = TileTensor(layers[i].mirror_posts.unsafe_ptr().unsafe_offset(group*rows*width),row_major((rows,width)))
                    matmul[use_tf32=False](ctx,reflected_target,lhs,reflected_rhs,c_row_major=True)
            if i == 0:
                ctx.enqueue_function[select_warp[False]](fv(layers[i].posts),rv(layers[i].mirror_posts),fv(layers[i].orientation),
                    Int64(width),Int64(py=c["winners"][i]),Int64(10*rows),grid_dim=ceildiv(10*rows,4),block_dim=128)
            else:
                ctx.enqueue_function[select_warp[True]](fv(layers[i].posts),rv(layers[i].mirror_posts),fv(layers[i].orientation),
                    Int64(width),Int64(py=c["winners"][i]),Int64(10*rows),grid_dim=ceildiv(10*rows,4),block_dim=128)
            ctx.enqueue_function[column_scores](rv(layers[i].posts),fv(energy),fv(health),Int64(size),Int64(locations),
                Int64(width),Int64(start),Int64(depth),Int64(i),grid_dim=10*size,block_dim=256)
            ctx.enqueue_function[observations](rv(layers[i].posts),fv(layers[i].nq),Int64(width),Int64(locations),Int64(size),grid_dim=ceildiv(10*size*width,128),block_dim=128)
            ctx.enqueue_function[utilization](rv(layers[i].nq),fv(layers[i].seen),fv(layers[i].positive_seen),iv(labels),Int64(size),Int64(width),Int64(start),Int64(1 if train else 0),grid_dim=ceildiv(10*width,128),block_dim=128)
            if Bool(py=c["pool_after"][i]):
                ctx.enqueue_function[antialias_pool](rv(layers[i].posts),fv(layers[i].pooled),Int64(layers[i].side),
                    Int64(width),Int64(10*size),grid_dim=ceildiv(10*size*(layers[i].side//2)*(layers[i].side//2)*width,128),block_dim=128)
            if trace:
                download(layers[i].inputs,packet["records"][i]["u"])
                download(layers[i].posts,packet["records"][i]["h"])
                download(layers[i].orientation,packet["records"][i]["orientation"])
        if train:
            ctx.enqueue_function[choose_margin_columns](rv(energy),iv(labels),rv(weights),fv(scales),Int64(start),Int64(size),Int64(depth),
                Int64(2),Int64(1),Float32(1),Int64(0),Float32(py=c["margin"]),grid_dim=ceildiv(size,128),block_dim=128)
            ctx.enqueue_function[compact_pairs](rv(scales),jv(pairs),jv(pair_counts),Int64(size),Int64(start),grid_dim=10,block_dim=128)
            for i in range(depth):
                var di, width, locations = layers[i].incoming,layers[i].width,layers[i].side*layers[i].side
                var splits = Int(py=packet["splits"][i])
                ctx.enqueue_function[batch_observations](rv(layers[i].nq),rv(scales),fv(layers[i].controls),
                    Int64(width),Int64(size),Int64(start),Int64(1),grid_dim=(width,10),block_dim=128)
                if i == 0:
                    ctx.enqueue_function[batch_coactivity[True]](rv(layers[i].inputs),rv(layers[i].posts),rv(layers[i].orientation),
                        rv(layers[i].nq),rv(scales),iv(pairs),iv(pair_counts),fv(layers[i].raw),Int64(di),Int64(width),
                        Int64(locations),Int64(size),Int64(start),Int64(py=c["kernels"][i]),Int64(splits),Int64(1),
                        grid_dim=(ceildiv(width,32),ceildiv(di,32),10*splits),block_dim=(16,16))
                else:
                    ctx.enqueue_function[compact_selected_observations](rv(layers[i].posts),rv(layers[i].orientation),rv(layers[i].nq),rv(scales),
                        iv(pairs),iv(pair_counts),jv(layers[i].selected_indices),fv(layers[i].selected_values),jv(layers[i].selected_counts),
                        Int64(width),Int64(locations),Int64(size),Int64(start),Int64(splits),grid_dim=(ceildiv(width,128),10*splits),block_dim=128)
                    ctx.enqueue_function[sparse_coactivity](rv(layers[i].inputs),iv(layers[i].selected_indices),rv(layers[i].selected_values),iv(layers[i].selected_counts),
                        rv(layers[i].controls),fv(layers[i].raw),Int64(di),Int64(width),Int64(locations),Int64(size),Int64(py=c["kernels"][i]),Int64(splits),
                        grid_dim=(ceildiv(di,1024),width,10*splits),block_dim=128)

            for index in range(depth):
                var i = depth-1-index if Bool(py=packet["reverse"]) else index
                ctx.enqueue_function[apply_batch_lion](fv(layers[i].w),fv(layers[i].m),jv(layers[i].clock),
                    rv(layers[i].raw),rv(layers[i].controls),fv(layers[i].numerator),fv(health),
                    Int64(layers[i].incoming),Int64(layers[i].width),Int64(py=packet["splits"][i]),
                    Float32(py=c["rate"]),Int64(1 if trace else 0),Float32(py=c["direction_history"]),
                    Float32(1.0-Float64(py=c["direction_history"])),Float32(py=c["history_decay"]),
                    Float32(1.0-Float64(py=c["history_decay"])),grid_dim=ceildiv(10*layers[i].incoming*layers[i].width,128),block_dim=128)
                if trace:
                    download(layers[i].numerator,packet["pooled_statistics"][i]["numerator"])
                    download(layers[i].controls,packet["pooled_statistics"][i]["nq"])
    ctx.synchronize()
    download(energy,packet["energies"])
    download(health,packet["health"])
    if train:
        download(scales,packet["scales"])
    for i in range(depth):
        download(layers[i].seen,packet["utilization"][i])
        download(layers[i].positive_seen,packet["positive_utilization"][i])
        if train:
            download(layers[i].w,packet["state"][i]["w"])
            download(layers[i].m,packet["state"][i]["m"])
            download(layers[i].clock,packet["state"][i]["clock"])
    return packet


@export
def PyInit_competitive_conv_connected_sparse_native() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("competitive_conv_connected_sparse_native")
        module.def_function[run]("run")
        return module.finalize()
    except error:
        abort(String(error))
