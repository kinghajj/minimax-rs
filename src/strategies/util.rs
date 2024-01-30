use super::super::interface::*;

use rand::Rng;

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
pub(super) fn move_id<G: Game>(s: &<G as Game>::S, m: Option<<G as Game>::M>) -> String {
    if let Some(mov) = m {
        G::notation(s, mov).unwrap_or("no notation impl".to_string())
    } else {
        "none".to_string()
    }
}

pub(super) fn pv_string<G: Game>(path: &[<G as Game>::M], state: &<G as Game>::S) -> String
where
    <G as Game>::M: Copy,
    <G as Game>::S: Clone,
{
    let mut state = state.clone();
    let mut out = String::new();
    for (i, &m) in (0..).zip(path.iter()) {
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(move_id::<G>(&state, Some(m)).as_str());
        if let Some(new_state) = G::apply(&mut state, &m) {
            state = new_state;
        }
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

static PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

// Find and return the highest scoring element of the set.
// If multiple elements have the highest score, select one randomly.
// Constraints:
//   - Don't call the scoring function more than once per element.
//   - Select one uniformly, so that a run of high scores doesn't
//       bias towards the one that scans first.
//   - Don't shuffle the input or allocate a new array for shuffling.
//   - Optimized for sets with <10k values.
pub(super) fn random_best<T, F: Fn(&T) -> f32>(set: &[T], score_fn: F) -> Option<&T> {
    // To make the choice more uniformly random among the best moves,
    // start at a random offset and stride by a random amount.
    // The stride must be coprime with n, so pick from a set of 5 digit primes.

    let n = set.len();
    // Combine both random numbers into a single rng call.
    let r = rand::thread_rng().gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let mut best_score = f32::NEG_INFINITY;
    let mut best = None;
    for _ in 0..n {
        let score = score_fn(&set[i]);
        debug_assert!(!score.is_nan());
        if score > best_score {
            best_score = score;
            best = Some(&set[i]);
        }
        i = (i + stride) % n;
    }
    best
}
