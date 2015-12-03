//! An implementation of Negamax.
//!
//! Currently, only the basic alpha-pruning variant is implemented. Further work
//! could add advanced features, like history and/or transposition tables. This
//! picks randomly among the "best" moves, so that it's non-deterministic.

use super::super::interface::*;
use scoped_threadpool::Pool;
use std::cmp::max;
use std::marker::PhantomData;

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
            break;
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
    _eval: PhantomData<E>,
}

impl<E: Evaluator> Negamax<E> {
    pub fn new(opts: Options) -> Self {
        Negamax {
            opts: opts,
            _eval: PhantomData,
        }
    }
}

impl<E: Evaluator> Grader<E::G> for Negamax<E>
    where <E::G as Game>::S: Clone,
          <E::G as Game>::M: Copy
{
    fn grade(&mut self, s: &<E::G as Game>::S, p: Player) -> Vec<Grade<<E::G as Game>::M>> {
        let mut moves = [None; 100];
        let num_moves = E::G::generate_moves(s, p, &mut moves);
        let mut s_clone = s.clone();
        moves.into_iter()
             .take(num_moves)
             .map(|m| m.unwrap())
             .map(|m| {
                 m.apply(&mut s_clone);
                 let value = -negamax::<E>(&mut s_clone,
                                           self.opts.max_depth,
                                           Evaluation::Worst,
                                           Evaluation::Best,
                                           -p);
                 m.undo(&mut s_clone);
                 Grade {
                     value: value,
                     play: m,
                 }
             })
             .collect()
    }
}

pub struct ParallelNegamax<E> {
    opts: Options,
    pool: Pool,
    _eval: PhantomData<E>,
}

impl<E: Evaluator> ParallelNegamax<E> {
    pub fn new(threads: u32, opts: Options) -> Self {
        ParallelNegamax {
            opts: opts,
            pool: Pool::new(threads),
            _eval: PhantomData,
        }
    }
}

impl<E: Evaluator> Grader<E::G> for ParallelNegamax<E>
    where <E::G as Game>::S: Clone + Send,
          <E::G as Game>::M: Copy + Send
{
    fn grade(&mut self, s: &<E::G as Game>::S, p: Player) -> Vec<Grade<<E::G as Game>::M>> {
        let mut moves = [None; 100];
        let num_moves = E::G::generate_moves(s, p, &mut moves);
        let max_depth = self.opts.max_depth;
        par_map_collect!(
            self.pool,
            moves.into_iter()
                 .take(num_moves)
                 .map(|m| (m.unwrap(), s.clone())),
            num_moves,
            (m, mut s) => {
                m.apply(&mut s);
                Grade {
                    value: -negamax::<E>(&mut s,
                                         max_depth,
                                         Evaluation::Worst,
                                         Evaluation::Best,
                                         -p),
                    play: m,
                }
             }
        )
    }
}
