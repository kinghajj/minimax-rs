//! Utility functions for testing, and tests.

use super::interface;
use super::interface::{Game, Move};
use std::default::Default;
use std::time::Instant;

/// Play a complete, new game with players using the two provided strategies.
///
/// Returns `None` if the game ends in a draw, or `Some(0)`, `Some(1)` if the
/// first or second strategy won, respectively.
pub fn battle_royale<G, S1, S2>(s1: &mut S1, s2: &mut S2) -> Option<usize>
where
    G: interface::Game,
    G::S: Default,
    S1: interface::Strategy<G>,
    S2: interface::Strategy<G>,
{
    let mut state = G::S::default();
    let mut strategies: [&mut dyn interface::Strategy<G>; 2] = [s1, s2];
    let mut s = 0;
    while G::get_winner(&state).is_none() {
        let ref mut strategy = strategies[s];
        match strategy.choose_move(&mut state) {
            Some(m) => m.apply(&mut state),
            None => break,
        }
        s = 1 - s;
    }
    match G::get_winner(&state).unwrap() {
        interface::Winner::Draw => None,
        interface::Winner::PlayerJustMoved => Some(1 - s),
        interface::Winner::PlayerToMove => Some(s),
    }
}

pub(crate) struct MovePool<M> {
    pool: Vec<Vec<M>>,
}

impl<M> Default for MovePool<M> {
    fn default() -> Self {
        Self { pool: Vec::new() }
    }
}

impl<M> MovePool<M> {
    pub(crate) fn new(&mut self) -> Vec<M> {
        self.pool.pop().unwrap_or_else(|| Vec::new())
    }

    pub(crate) fn free(&mut self, mut vec: Vec<M>) {
        vec.clear();
        self.pool.push(vec);
    }
}

fn perft_recurse<G: Game>(pool: &mut MovePool<G::M>, state: &mut G::S, depth: usize) -> u64 {
    if depth == 0 {
        return 1;
    }
    if G::get_winner(state).is_some() {
        // Apparently perft rules only count positions at the target depth.
        return 0;
    }
    let mut moves = pool.new();
    G::generate_moves(state, &mut moves);
    let n = if depth == 1 {
        moves.len() as u64
    } else {
        let mut count = 0;
        for m in moves.iter() {
            m.apply(state);
            count += perft_recurse::<G>(pool, state, depth - 1);
            m.undo(state);
        }
        count
    };
    pool.free(moves);
    n
}

pub fn perft<G: Game>(state: &mut <G as Game>::S, max_depth: usize) -> Vec<u64> {
    println!("depth           count        time        kn/s");
    let mut pool = MovePool::<G::M>::default();
    let mut counts = Vec::new();
    for depth in 0..max_depth + 1 {
        let start = Instant::now();
        let count = perft_recurse::<G>(&mut pool, state, depth);
        let dur = start.elapsed();
        let rate = count as f64 / dur.as_secs_f64() / 1000.0;
        println!("{:>5} {:>15} {:>11} {:>11.1}", depth, count, format!("{:.1?}", dur), rate);
        counts.push(count);
    }
    counts
}
