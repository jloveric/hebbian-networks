"""Fixed binomial filtering of stride-one signed magnitude pooling."""

from std.math import abs
from max.gpu import global_idx
from gpu_buffer_utils import F, R


@inline(.always)
def reflected_coordinate(coordinate: Int, side: Int) -> Int:
    if side == 1:
        return 0
    if coordinate < 0:
        return -coordinate
    if coordinate >= side:
        return 2 * side - 2 - coordinate
    return coordinate


def antialias_pool(
    source: R, output: F, side_arg: Int64,
    channels_arg: Int64, paths_arg: Int64,
):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert source.flat_rank == output.flat_rank == 1
    var at = global_idx.x
    var side, channels = Int(side_arg), Int(channels_arg)
    var small = side // 2
    if at >= Int(paths_arg) * small * small * channels:
        return
    var cell, channel = divmod(at, channels)
    var path, location = divmod(cell, small * small)
    var y, x = divmod(location, small)
    var total: Float32 = 0
    # Fuse the dense max and fixed blur to avoid an intermediate feature map.
    comptime for ky in range(3):
        comptime for kx in range(3):
            var py = reflected_coordinate(2 * y + ky - 1, side - 1)
            var px = reflected_coordinate(2 * x + kx - 1, side - 1)
            var best: Float32 = 0
            comptime for dy in range(2):
                comptime for dx in range(2):
                    var value = source[((path * side + py + dy) * side + px + dx) * channels + channel]
                    if abs(value) > abs(best) or (abs(value) == abs(best) and value > best):
                        best = value
            comptime weight = Float32((2 if ky == 1 else 1) * (2 if kx == 1 else 1)) / 16
            total += best * weight
    output[at] = total
