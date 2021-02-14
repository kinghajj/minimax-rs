//! Utility functions for testing, and tests.

use super::interface;
use super::interface::Move;
use std::default::Default;

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
