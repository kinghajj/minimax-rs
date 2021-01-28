//! An implementation of Negamax.
//!
//! Currently, only the basic alpha-pruning variant is implemented. Further work
//! could add advanced features, like history and/or transposition tables. This
//! picks randomly among the "best" moves, so that it's non-deterministic.

use super::super::interface::*;
use rand;
use rand::Rng;
use core::cmp::max;
use core::marker::PhantomData;

fn negamax<E: Evaluator>(s: &mut <E::G as Game>::S,
                         depth: usize,
                         mut alpha: Evaluation,
                         beta: Evaluation,
                         p: Player)
                         -> Evaluation
    where <<E as Evaluator>::G as Game>::M: Copy
{
    let maybe_winner = E::G::get_winner(s);
    if depth == 0 || maybe_winner.is_some() {
        return p * E::evaluate(s, maybe_winner);
    }
    let mut moves = [None; 100];
    E::G::generate_moves(s, p, &mut moves);
    let mut best = Evaluation::Worst;
    for m in moves.iter().take_while(|om| om.is_some()).map(|om| om.unwrap()) {
        m.apply(s);
        let value = -negamax::<E>(s, depth - 1, -beta, -alpha, -p);
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
    fn choose_move(&mut self, s: &<E::G as Game>::S, p: Player) -> Option<<E::G as Game>::M> {
        let mut best = Evaluation::Worst;
        let mut moves = [None; 100];
        E::G::generate_moves(s, p, &mut moves);
        let mut candidate_moves = [false; 100];
        let mut s_clone = s.clone();
	for (m,i) in moves.iter().take_while(|m| m.is_some()).map(|m| m.unwrap()).zip(0..) {
            // determine value for this move
            m.apply(&mut s_clone);
            let value = -negamax::<E>(&mut s_clone,
                                      self.opts.max_depth,
                                      Evaluation::Worst,
                                      Evaluation::Best,
                                      -p);
            m.undo(&mut s_clone);
            // this move is a candidate move
            if value == best {
                candidate_moves[i] = true;
            // this move is better than any previous, so it's the sole candidate
            } else if value > best {
                candidate_moves = [false; 100];
                candidate_moves[i] = true;
                best = value;
            }
        }
	let i = self.rng.gen_range(0, candidate_moves.iter().filter(|&&x|x).count());
	*moves.iter().zip(&candidate_moves).filter(|(_,&b)|b).nth(i)?.0
    }
}
