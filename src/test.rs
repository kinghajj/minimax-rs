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
    let mut strategies: Vec<(interface::Player, &mut interface::Strategy<G>)> = vec![
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

// Ensure that two players using negamax always results in a draw.
#[test]
fn test_ttt_negamax_always_draws() {
    use super::ttt;
    use super::strategies::negamax::{Negamax, Options};
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    for _ in 0..100 {
        assert!(battle_royale(&mut s1, &mut s2) == interface::Winner::Draw)
    }
}

// Ensure that a player using negamax against a random one always results in
// either a draw or a win for the former player.
#[test]
fn test_ttt_negamax_vs_random_always_wins_or_draws() {
    use super::ttt;
    use super::strategies::negamax::{Negamax, Options};
    use super::strategies::random::Random;
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Random::new();
    for _ in 0..100 {
        assert!(battle_royale(&mut s1, &mut s2) !=
                interface::Winner::Competitor(interface::Player::Opponent))
    }
}
