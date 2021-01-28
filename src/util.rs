//! Utility functions for testing, and tests.

use super::interface;
use super::interface::Move;
use std::default::Default;

/// Play a complete, new game with players using the two provided strategies.
///
/// The first strategy will be `Player::Computer`, the other `Player::Opponent`.
/// Returns result of the game.
pub fn battle_royale<G, S1, S2>(s1: &mut S1, s2: &mut S2) -> interface::Winner
    where G: interface::Game,
          G::S: Default,
          S1: interface::Strategy<G>,
          S2: interface::Strategy<G>
{
    let mut state = G::S::default();
    let mut strategies: [(interface::Player, &mut dyn interface::Strategy<G>); 2] = [
            (interface::Player::Computer, s1),
            (interface::Player::Opponent, s2),
        ];
    let mut s = 0;
    while G::get_winner(&state).is_none() {
        let (p, ref mut strategy) = strategies[s];
        match strategy.choose_move(&mut state, p) {
            Some(m) => m.apply(&mut state),
            None => break,
        }
        s = 1 - s;
    }
    G::get_winner(&state).unwrap()
}
