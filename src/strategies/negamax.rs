//! An implementation of Negamax.
//!
//! With only the basic alpha-pruning implemented. This picks randomly among
//! the "best" moves, so that it's non-deterministic.

use super::super::interface::*;
use rand;
use rand::Rng;
use std::cmp::max;
use std::marker::PhantomData;

// For values near winning and losing values, push them slightly closer to zero.
// A win in 3 moves (BEST-3) will be chosen over a win in 5 moves (BEST-5).
// A loss in 5 moves (WORST+5) will be chosen over a loss in 3 moves (WORST+3).
fn degrade_wins(value: Evaluation) -> Evaluation {
    if value > BEST_EVAL - 100 {
        value - 1
    } else if value < WORST_EVAL + 100 {
        value + 1
    } else {
        value
    }
}

fn negamax<E: Evaluator>(
    s: &mut <E::G as Game>::S, depth: usize, mut alpha: Evaluation, beta: Evaluation,
) -> Evaluation
where
    <<E as Evaluator>::G as Game>::M: Copy,
{
    if let Some(winner) = E::G::get_winner(s) {
        return winner.evaluate();
    }
    if depth == 0 {
        return E::evaluate(s);
    }
    let mut moves = [None; 200];
    E::G::generate_moves(s, &mut moves);
    let mut best = WORST_EVAL;
    for m in moves.iter().take_while(|om| om.is_some()).map(|om| om.unwrap()) {
        m.apply(s);
        let value = -negamax::<E>(s, depth - 1, -beta, -alpha);
        m.undo(s);
        best = max(best, value);
        alpha = max(alpha, value);
        if alpha >= beta {
            break;
        }
    }
    degrade_wins(best)
}

/// Options to use for the `Negamax` engine.
pub struct Options {
    /// The maximum depth within the game tree.
    pub max_depth: usize,
}

pub struct Negamax<E> {
    opts: Options,
    rng: rand::ThreadRng,
    prev_value: Evaluation,
    _eval: PhantomData<E>,
}

impl<E: Evaluator> Negamax<E> {
    pub fn new(opts: Options) -> Negamax<E> {
        Negamax { opts: opts, rng: rand::thread_rng(), prev_value: 0, _eval: PhantomData }
    }

    // Return the value computed for the root node for the last computation.
    pub fn root_value(&self) -> Evaluation {
        // Undo any value clamping.
        if self.prev_value > BEST_EVAL - 100 {
            BEST_EVAL
        } else if self.prev_value < WORST_EVAL + 100 {
            WORST_EVAL
        } else {
            self.prev_value
        }
    }
}

impl<E: Evaluator> Strategy<E::G> for Negamax<E>
where
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        let mut best = WORST_EVAL;
        let mut moves = [None; 200];
        let n = E::G::generate_moves(s, &mut moves);
        // Randomly permute order that we look at the moves.
        // We'll pick the first best score from this list.
        self.rng.shuffle(&mut moves[..n]);

        let mut best_move = moves[0]?;
        let mut s_clone = s.clone();
        for m in moves.iter().take_while(|m| m.is_some()).map(|m| m.unwrap()) {
            // determine value for this move
            m.apply(&mut s_clone);
            let value = -negamax::<E>(&mut s_clone, self.opts.max_depth, WORST_EVAL, -best);
            m.undo(&mut s_clone);
            // Strictly better than any move found so far.
            if value > best {
                best = value;
                best_move = m;
            }
        }
        self.prev_value = best;
        Some(best_move)
    }
}
