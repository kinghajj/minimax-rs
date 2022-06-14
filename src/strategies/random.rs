//! A strategy that randomly chooses a move, for use in tests.

use super::super::interface::*;
use rand::seq::SliceRandom;
use std::marker::PhantomData;

pub struct Random<G: Game> {
    rng: rand::rngs::ThreadRng,
    game_type: PhantomData<G>,
}

impl<G: Game> Random<G> {
    pub fn new() -> Self {
        Self { rng: rand::thread_rng(), game_type: PhantomData }
    }
}

impl<G: Game> Default for Random<G> {
    fn default() -> Self {
        Random::new()
    }
}

impl<G: Game> Strategy<G> for Random<G>
where
    G::M: Copy,
{
    fn choose_move(&mut self, s: &G::S) -> Option<G::M> {
        let mut moves = Vec::new();
        G::generate_moves(s, &mut moves);
        moves.choose(&mut self.rng).copied()
    }
}
