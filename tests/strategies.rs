// For a given Evaluator and depth, all Strategies should produce the same
// value for the root. They use different techniques and pruning heuristics
// for speed, but it's all fundamentally the minimax algorithm. This file
// creates fake evaluation trees of connect four, and ensures that all
// Strategies (including a plain negamax without alpha-beta) get the same answer.

extern crate minimax;
extern crate rand;
#[path = "../examples/connect4.rs"]
mod connect4;

use minimax::interface::*;
use rand::Rng;
use std::cmp::max;
use std::marker::PhantomData;

fn negamax<E: Evaluator>(s: &mut <E::G as Game>::S, depth: usize) -> Evaluation
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
    let n = E::G::generate_moves(s, &mut moves);
    let mut best = WORST_EVAL;
    for m in moves[..n].iter().map(|m| m.unwrap()) {
        m.apply(s);
        let value = -negamax::<E>(s, depth - 1);
        m.undo(s);
        best = max(best, value);
    }
    best
}

pub struct PlainNegamax<E> {
    depth: usize,
    root_value: Evaluation,
    _eval: PhantomData<E>,
}

impl<E: Evaluator> PlainNegamax<E> {
    pub fn new(depth: usize) -> PlainNegamax<E> {
        PlainNegamax { depth: depth, root_value: 0, _eval: PhantomData }
    }
}

impl<E: Evaluator> Strategy<E::G> for PlainNegamax<E>
where
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        let mut moves = [None; 200];
        let n = E::G::generate_moves(s, &mut moves);

        let mut best_move = None;
        let mut best_value = WORST_EVAL;
        let mut s_clone = s.clone();
        for m in moves[..n].iter().map(|m| m.unwrap()) {
            m.apply(&mut s_clone);
            let value = -negamax::<E>(&mut s_clone, self.depth);
            m.undo(&mut s_clone);
            if value > best_value {
                best_value = value;
                best_move = Some(m);
            }
        }
        self.root_value = best_value;
        best_move
    }
}

struct RandomEvaluator;

impl minimax::Evaluator for RandomEvaluator {
    type G = connect4::Game;
    fn evaluate(b: &connect4::Board) -> minimax::Evaluation {
        // Scramble the game state to get a deterministically random Evaluation.
        let mut hash = b.pieces_just_moved().wrapping_mul(0xe512dc15f0da3dd1);
        hash = hash
            .wrapping_add(hash >> 33)
            .wrapping_add(b.pieces_to_move)
            .wrapping_mul(0x18d9db91aa689617);
        hash = hash.wrapping_add(hash >> 31);
        // Use fewer bits so that we get some equal values.
        (hash as minimax::Evaluation) >> 25
    }
}

fn generate_random_state(depth: usize) -> connect4::Board {
    let mut rng = rand::thread_rng();
    let mut b = connect4::Board::default();
    for _ in 0..depth {
        let mut moves = [None; 10];
        let n = connect4::Game::generate_moves(&b, &mut moves);
        let m = moves[rng.gen_range(0, n)].unwrap();
        m.apply(&mut b);
        if connect4::Game::get_winner(&b).is_some() {
            // Oops, undo and try again on the next iter.
            m.undo(&mut b);
        }
    }
    b
}

#[test]
fn compare_plain_negamax() {
    for _ in 0..10 {
        for max_depth in 0..5 {
            let b = generate_random_state(10);

            let mut plain_negamax = PlainNegamax::<RandomEvaluator>::new(max_depth);
            plain_negamax.choose_move(&b);
            let value = plain_negamax.root_value;

            let mut negamax = minimax::Negamax::<RandomEvaluator>::with_max_depth(max_depth);
            negamax.choose_move(&b);
            let negamax_value = negamax.root_value();
            assert_eq!(value, negamax_value, "search depth={}\n{}", max_depth, b);

            for &strategy in &[
                minimax::Replacement::Always,
                minimax::Replacement::DepthPreferred,
                minimax::Replacement::TwoTier,
            ] {
                let mut iterative = minimax::IterativeSearch::<RandomEvaluator>::new(
                    minimax::IterativeOptions::new()
                        .with_table_byte_size(64000)
                        .with_replacement_strategy(strategy)
                        .with_null_window_search(true),
                );
                iterative.set_max_depth(max_depth);
                iterative.choose_move(&b);
                let iterative_value = iterative.root_value();
                assert_eq!(
                    value, iterative_value,
                    "search depth={}, strategy={:?}\n{}",
                    max_depth, strategy, b
                );
            }
        }
    }
}
