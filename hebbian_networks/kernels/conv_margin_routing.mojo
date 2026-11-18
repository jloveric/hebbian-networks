"""The declared normalized margin controls eligibility; unit pair weights are unchanged."""
from max.gpu import global_idx
from gpu_buffer_utils import F,R,I


def choose_margin_columns_for_columns[columns: Int = 10](energy: R, labels: I, weights: R, scales: F,
                            start_arg: Int64, size_arg: Int64, depth_arg: Int64,
                            policy_arg: Int64, combined_arg: Int64, negative_scale: Float32, always_arg: Int64, margin: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    comptime assert energy.flat_rank == labels.flat_rank == weights.flat_rank == scales.flat_rank == 1
    var image = Int(start_arg)+global_idx.x
    if global_idx.x >= Int(size_arg):
        return
    var depth, label = Int(depth_arg), Int(labels[image])
    var scores = SIMD[DType.float32,128 if columns > 16 else 16](0)
    var predicted, rival = 0, -1
    for k in range(columns):
        if combined_arg != 0:
            for layer in range(depth):
                scores[k] = scores[k]+energy[(image*columns+k)*depth+layer]*weights[layer]
        else:
            scores[k] = energy[(image*columns+k)*depth+depth-1]
        if scores[k] > scores[predicted]:
            predicted = k
        if k != label and (rival < 0 or scores[k] > scores[rival]):
            rival = k
    var good,bad = scores[label],scores[rival]
    var included = always_arg != 0 or policy_arg == 0 or predicted != label or (good-bad)/(good+bad+Float32(1e-30)) < margin
    for k in range(columns):
        var value = Float32(0)
        if included:
            if k == label:
                value = 1
            elif policy_arg == 2:
                value = -1 if k == rival else 0
            else:
                value = -negative_scale
        scales[image*columns+k] = value



# Keep the original non-parametric kernel entry point for ten-class readers.
comptime choose_margin_columns = choose_margin_columns_for_columns[10]
