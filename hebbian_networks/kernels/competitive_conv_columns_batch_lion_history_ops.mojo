"""One pooled local Lion step with explicit momentum conventions."""
from std.math import sqrt, fma, isfinite
from max.gpu import global_idx
from competitive_gru import F, R, J


def apply_batch_lion_for_columns[columns: Int = 10](weights: F, history: F, clocks: J, partials: R, controls: R,
        numerator: F, health: F, incoming_arg: Int64, width_arg: Int64,
        splits_arg: Int64, rate: Float32, trace_arg: Int64,
        direction_history: Float32, direction_current: Float32,
        history_decay: Float32, history_current: Float32):
    comptime MAX_THREADS_PER_BLOCK_METADATA = 128
    var at = global_idx.x
    var incoming,width,splits = Int(incoming_arg),Int(width_arg),Int(splits_arg)
    if at >= columns*incoming*width:
        return
    var column,coordinate = divmod(at,incoming*width)
    var j = coordinate%width
    var n = controls[2*(column*width+j)]
    if n == 0:
        if trace_arg != 0:
            numerator[at] = 0
        return
    var total = Float32(0)
    for part in range(splits):
        total += partials[(column*splits+part)*incoming*width+coordinate]
    if trace_arg != 0:
        numerator[at] = total
    var a,q = total/n,controls[2*(column*width+j)+1]/n
    var old_m = history[at]
    var signal = fma(direction_current,a,direction_history*old_m)
    var direction = Float32(1 if signal > 0 else -1 if signal < 0 else 0)/sqrt(Float32(incoming))
    var weight = (weights[at]+rate*direction)/(1+rate*q)
    var next_m = fma(history_current,a,history_decay*old_m)
    weights[at],history[at] = weight,next_m
    if coordinate < width:
        clocks[column*width+j] += 1
    if not isfinite(weight) or not isfinite(next_m) or not isfinite(q):
        health[0] = 0


# Keep the original non-parametric kernel entry point for ten-class readers.
comptime apply_batch_lion = apply_batch_lion_for_columns[10]
