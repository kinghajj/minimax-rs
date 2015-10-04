//! Basic graded move choosers.

use super::interface::*;
use rand::Rng;

/// Always chooses the first, best move from a set.
pub struct Deterministic;

impl<M: Clone> Chooser<M> for Deterministic {
    fn choose(&mut self, evaluated_moves: &[Grade<M>]) -> Option<M> {
        let mut best = Evaluation::Worst;
        let mut candidates = Vec::new();
        for &Grade { value: e, play: ref m } in evaluated_moves.iter() {
            if e == best {
                candidates.push(m.clone())
            } else if e > best {
                candidates.clear();
                candidates.push(m.clone());
                best = e
            }
        }
        if candidates.is_empty() {
            None
        } else {
            Some(candidates.into_iter().next().unwrap())
        }
    }
}

/// Chooses randomly from the best moves from a set.
pub struct Random<R> {
    rng: R,
}

impl<R: Rng> Random<R> {
    pub fn new(rng: R) -> Self {
        Random { rng: rng }
    }
}

impl<M: Clone, R: Rng> Chooser<M> for Random<R> {
    fn choose(&mut self, evaluated_moves: &[Grade<M>]) -> Option<M> {
        let mut best = Evaluation::Worst;
        let mut candidates = Vec::new();
        for &Grade { value: e, play: ref m } in evaluated_moves.iter() {
            if e == best {
                candidates.push(m.clone())
            } else if e > best {
                candidates.clear();
                candidates.push(m.clone());
                best = e
            }
        }
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[self.rng.gen_range(0, candidates.len())].clone())
        }
    }
}
