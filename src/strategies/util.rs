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

// Return a unique id for humans for this move.
pub(super) fn move_id<G: Game>(s: &mut <G as Game>::S, m: Option<<G as Game>::M>) -> String
where
    <G as Game>::S: Zobrist,
{
    if let Some(mov) = m {
        if let Some(notation) = mov.notation(s) {
            notation
        } else {
            mov.apply(s);
            let id = format!("{:06x}", s.zobrist_hash() & 0xffffff);
            mov.undo(s);
            id
        }
    } else {
        "none".to_string()
    }
}

pub(super) fn pv_string<G: Game>(path: &[<G as Game>::M], s: &mut <G as Game>::S) -> String
where
    <G as Game>::S: Zobrist,
    <G as Game>::M: Copy,
{
    let mut out = String::new();
    for (i, m) in (0..).zip(path.iter()) {
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(move_id::<G>(s, Some(*m)).as_str());
        m.apply(s);
    }
    for m in path.iter().rev() {
        m.undo(s);
    }
    out
}

pub(super) fn move_to_front<M: Eq>(m: M, moves: &mut [M]) {
    for i in 0..moves.len() {
        if moves[i] == m {
            moves[0..i + 1].rotate_right(1);
            break;
        }
    }
}

// This exists to be wrapped in a mutex, because it didn't work when I tried a tuple.
pub(super) struct ValueMove<M> {
    pub(super) value: Evaluation,
    pub(super) m: M,
}

impl<M> ValueMove<M> {
    pub(super) fn new(value: Evaluation, m: M) -> Self {
        Self { value, m }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn max(&mut self, value: Evaluation, m: M) {
        if value > self.value {
            self.value = value;
            self.m = m;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn into_inner(self) -> (Evaluation, M) {
        (self.value, self.m)
    }
}
