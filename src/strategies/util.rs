use super::super::interface::*;

// For values near winning and losing values, push them slightly closer to zero.
// A win in 3 moves (BEST-3) will be chosen over a win in 5 moves (BEST-5).
// A loss in 5 moves (WORST+5) will be chosen over a loss in 3 moves (WORST+3).
pub(super) fn clamp_value(value: Evaluation) -> Evaluation {
    if value > BEST_EVAL - 100 {
        value - 1
    } else if value < WORST_EVAL + 100 {
        value + 1
    } else {
        value
    }
}

// Undo any value clamping.
pub(super) fn unclamp_value(value: Evaluation) -> Evaluation {
    if value > BEST_EVAL - 100 {
        BEST_EVAL
    } else if value < WORST_EVAL + 100 {
        WORST_EVAL
    } else {
        value
    }
}
