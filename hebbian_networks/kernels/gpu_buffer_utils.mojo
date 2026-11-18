"""GPU buffer utilities extracted from the historical source module.

Only the image models' imported types and copy/view helpers are retained.
"""
from std.python import PythonObject
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

comptime Layout = type_of(row_major((Int(0),)))
comptime F = TileTensor[DType.float32, Layout, MutAnyOrigin]
comptime R = TileTensor[DType.float32, Layout, ImmutAnyOrigin]
comptime J = TileTensor[DType.int64, Layout, MutAnyOrigin]
comptime I = TileTensor[DType.int64, Layout, ImmutAnyOrigin]


def fv(mut b: DeviceBuffer[DType.float32]) -> F:
    return TileTensor(b, row_major((len(b),)))


def rv(b: DeviceBuffer[DType.float32]) -> R:
    return TileTensor(b, row_major((len(b),))).as_imm()


def jv(mut b: DeviceBuffer[DType.int64]) -> J:
    return TileTensor(b, row_major((len(b),)))


def iv(b: DeviceBuffer[DType.int64]) -> I:
    return TileTensor(b, row_major((len(b),))).as_imm()


def upload[dtype: DType](ctx: DeviceContext, a: PythonObject) raises -> DeviceBuffer[dtype]:
    var size = Int(py=a.size)
    var source = Pointer[Scalar[dtype], ImmutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    var host = ctx.enqueue_create_host_buffer[dtype](size)
    for i in range(size):
        host[i] = source[unsafe_offset=i]
    var device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(device, host)
    return device^


def download[dtype: DType](buffer: DeviceBuffer[dtype], a: PythonObject) raises:
    var size = Int(py=a.size)
    if size > len(buffer):
        raise Error("download exceeds native buffer")
    var destination = Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    with buffer.map_to_host() as host:
        for i in range(size):
            destination[unsafe_offset=i] = host[i]
