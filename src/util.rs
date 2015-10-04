//! Utility functions for testing, and tests.

use super::interface;
use super::interface::Move;
use std::default::Default;

/// Play a complete, new game with players using the two provided strategies.
///
/// The first strategy will be `Player::Computer`, the other `Player::Opponent`.
/// Returns result of the game.
pub fn battle_royale<Game>(grader1: &mut interface::Grader<Game>,
                           chooser1: &mut interface::Chooser<Game::M>,
                           grader2: &mut interface::Grader<Game>,
                           chooser2: &mut interface::Chooser<Game::M>)
                           -> interface::Winner
    where Game: interface::Game,
          Game::S: Default
{
    let mut state = Game::S::default();
    let mut players: Vec<(_, &mut interface::Grader<Game>, &mut interface::Chooser<Game::M>)> =
        vec![
            (interface::Player::Computer, grader1, chooser1),
            (interface::Player::Opponent, grader2, chooser2),
        ];
    let mut s = 0;
    while Game::get_winner(&state).is_none() {
        let (p, ref mut grader, ref mut chooser) = players[s];
        let graded_moves = grader.grade(&state, p);
        match chooser.choose(&graded_moves) {
            None => break,
            Some(m) => m.apply(&mut state),
        }
        s = 1 - s
    }
    Game::get_winner(&state).unwrap()
}
