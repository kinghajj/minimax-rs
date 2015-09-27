//! A strategy that randomly chooses a move, for use in tests.

use super::super::interface::*;
use rand;
use rand::Rng;

pub struct Random { rng: rand::ThreadRng }

impl Random {
    pub fn new() -> Random {
        Random {
            rng: rand::thread_rng()
        }
    }
}

impl<G: Game> Strategy<G> for Random
    where G::M: Copy {
    fn choose_move(&mut self, s: &G::S, p: Player)
        -> Option<G::M> {
        let mut moves: [Option<G::M>; 100] = [None; 100];
        match G::generate_moves(s, p, &mut moves) {
            0 => None,
            num_moves => Some(moves[self.rng.gen_range(0, num_moves)].unwrap()),
        }
    }
}
