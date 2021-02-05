//! An implementation of Negamax.
//!
//! Currently, only the basic alpha-pruning variant is implemented. Further work
//! could add advanced features, like history and/or transposition tables. This
//! picks randomly among the "best" moves, so that it's non-deterministic.

use super::super::interface::*;
use rand;
use rand::Rng;
use std::cmp::max;
use std::marker::PhantomData;

fn negamax<E: Evaluator>(s: &mut <E::G as Game>::S,
                         depth: usize,
                         mut alpha: Evaluation,
                         beta: Evaluation)
                         -> Evaluation
    where <<E as Evaluator>::G as Game>::M: Copy
{
    if let Some(winner) = E::G::get_winner(s) {
	return winner.evaluate();
    }
    if depth == 0 {
        return E::evaluate(s);
    }
    let mut moves = [None; 200];
    E::G::generate_moves(s, &mut moves);
    let mut best = Evaluation::Worst;
    for m in moves.iter().take_while(|om| om.is_some()).map(|om| om.unwrap()) {
        m.apply(s);
        let value = -negamax::<E>(s, depth - 1, -beta, -alpha);
        m.undo(s);
        best = max(best, value);
        alpha = max(alpha, value);
        if alpha >= beta {
            break
        }
    }
    best
}

/// Options to use for the `Negamax` engine.
pub struct Options {
    /// The maximum depth within the game tree.
    pub max_depth: usize,
}

pub struct Negamax<E> {
    opts: Options,
    rng: rand::ThreadRng,
    _eval: PhantomData<E>,
}

impl<E: Evaluator> Negamax<E> {
    pub fn new(opts: Options) -> Negamax<E> {
        Negamax {
            opts: opts,
            rng: rand::thread_rng(),
            _eval: PhantomData,
        }
    }
}

impl<E: Evaluator> Strategy<E::G> for Negamax<E>
    where <E::G as Game>::S: Clone,
          <E::G as Game>::M: Copy {
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        let mut best = Evaluation::Worst;
        let mut moves = [None; 200];
        let n = E::G::generate_moves(s, &mut moves);
        // Randomly permute order that we look at the moves.
        // We'll pick the first best score from this list.
        self.rng.shuffle(&mut moves[..n]);

        let mut best_move = moves.iter().next()?.unwrap();
        let mut s_clone = s.clone();
        for m in moves.iter().take_while(|m| m.is_some()).map(|m| m.unwrap()) {
            // determine value for this move
            m.apply(&mut s_clone);
            let value = -negamax::<E>(&mut s_clone,
                                      self.opts.max_depth,
                                      Evaluation::Worst,
                                      -best);
            m.undo(&mut s_clone);
            // Strictly better than any move found so far.
            if value > best {
                best = value;
                best_move = m;
            }
        }
        Some(best_move)
    }
}
