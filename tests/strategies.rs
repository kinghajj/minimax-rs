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
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

pub struct PlainNegamax<E: Evaluator> {
    depth: usize,
    root_value: Evaluation,
    // All moves tied with the best valuation.
    best_moves: Vec<<E::G as Game>::M>,
    eval: E,
}

impl<E: Evaluator> PlainNegamax<E> {
    pub fn new(eval: E, depth: usize) -> PlainNegamax<E> {
        PlainNegamax { depth: depth, root_value: 0, best_moves: Vec::new(), eval }
    }

    fn negamax(&self, s: &mut <E::G as Game>::S, depth: usize) -> Evaluation
    where
        <<E as Evaluator>::G as Game>::M: Copy,
    {
        if let Some(winner) = E::G::get_winner(s) {
            return winner.evaluate();
        }
        if depth == 0 {
            return self.eval.evaluate(s);
        }
        let mut moves = Vec::new();
        E::G::generate_moves(s, &mut moves);
        let mut best = WORST_EVAL;
        for m in moves.iter() {
            m.apply(s);
            let value = -self.negamax(s, depth - 1);
            m.undo(s);
            best = max(best, value);
        }
        best
    }
}

impl<E: Evaluator> Strategy<E::G> for PlainNegamax<E>
where
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        let mut moves = Vec::new();
        E::G::generate_moves(s, &mut moves);

        self.best_moves.clear();
        let mut best_value = WORST_EVAL;
        let mut s_clone = s.clone();
        for &m in moves.iter() {
            m.apply(&mut s_clone);
            let value = -self.negamax(&mut s_clone, self.depth);
            m.undo(&mut s_clone);
            if value == best_value {
                self.best_moves.push(m);
            } else if value > best_value {
                best_value = value;
                self.best_moves.clear();
                self.best_moves.push(m);
            }
        }
        self.root_value = best_value;
        self.best_moves.first().map(|m| *m)
    }
}

struct RandomEvaluator;

impl Default for RandomEvaluator {
    fn default() -> Self {
        Self {}
    }
}

impl minimax::Evaluator for RandomEvaluator {
    type G = connect4::Game;
    fn evaluate(&self, b: &connect4::Board) -> minimax::Evaluation {
        // Hash the game state to get a deterministically random Evaluation.
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(b.pieces_just_moved());
        hasher.write_u64(b.pieces_to_move);
        let hash = hasher.finish();
        // Use fewer bits so that we get some equal values.
        (hash as minimax::Evaluation) >> 25
    }
}

fn generate_random_state(depth: usize) -> connect4::Board {
    let mut rng = rand::thread_rng();
    let mut b = connect4::Board::default();
    for _ in 0..depth {
        let mut moves = Vec::new();
        connect4::Game::generate_moves(&b, &mut moves);
        let m = moves[rng.gen_range(0, moves.len())];
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
    for _ in 0..100 {
        for max_depth in 0..5 {
            let b = generate_random_state(10);

            let mut plain_negamax = PlainNegamax::new(RandomEvaluator::default(), max_depth);
            plain_negamax.choose_move(&b);
            let value = plain_negamax.root_value;

            let mut negamax = minimax::Negamax::new(RandomEvaluator, max_depth);
            let negamax_move = negamax.choose_move(&b).unwrap();
            let negamax_value = negamax.root_value();
            assert_eq!(value, negamax_value, "search depth={}\n{}", max_depth, b);
            assert!(
                plain_negamax.best_moves.contains(&negamax_move),
                "bad move={:?}\nsearch depth={}\n{}",
                negamax_move,
                max_depth,
                b
            );

            // Sampling of the configuration space.
            for (option_num, opt) in vec![
                minimax::IterativeOptions::new()
                    .with_replacement_strategy(minimax::Replacement::DepthPreferred)
                    .with_null_window_search(true),
                minimax::IterativeOptions::new()
                    .with_replacement_strategy(minimax::Replacement::Always)
                    .with_double_step_increment(),
                minimax::IterativeOptions::new()
                    .with_replacement_strategy(minimax::Replacement::TwoTier),
            ]
            .drain(..)
            .enumerate()
            {
                let mut iterative = minimax::IterativeSearch::new(
                    RandomEvaluator::default(),
                    opt.with_table_byte_size(64000),
                );
                iterative.set_max_depth(max_depth);
                let iterative_move = iterative.choose_move(&b).unwrap();
                let iterative_value = iterative.root_value();
                assert_eq!(
                    value, iterative_value,
                    "search depth={}, option={}\n{}",
                    max_depth, option_num, b
                );
                assert!(
                    plain_negamax.best_moves.contains(&iterative_move),
                    "bad move={:?}\nsearch depth={}\n{}",
                    iterative_move,
                    max_depth,
                    b
                );
            }
        }
    }
}
