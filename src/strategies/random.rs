//! A strategy that randomly chooses a move, for use in tests.

use super::super::interface::*;
use rand::seq::SliceRandom;

pub struct Random {
    rng: rand::rngs::ThreadRng,
}

impl Random {
    pub fn new() -> Random {
        Random { rng: rand::thread_rng() }
    }
}

impl Default for Random {
    fn default() -> Self {
        Random::new()
    }
}

impl<G: Game> Strategy<G> for Random
where
    G::M: Copy,
{
    fn choose_move(&mut self, s: &G::S) -> Option<G::M> {
        let mut moves = Vec::new();
        G::generate_moves(s, &mut moves);
        moves.choose(&mut self.rng).copied()
    }
}
