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
    let mut players: Vec<(_,
                          &mut interface::Grader<Game>,
                          &mut interface::Chooser<Game::M>)> = vec![
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

/// Use a `scoped_threadpool` to to apply a block `$f` to the `$len` `$t`-like
/// items of an iterator `$iter`, resulting in a vector of the results of `$f`.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate minimax;
/// # extern crate scoped_threadpool;
/// # fn main() {
/// let mut pool = scoped_threadpool::Pool::new(4);
/// let inputs = vec![1, 2, 3, 4];
/// assert_eq!(
///     par_map_collect!(pool, inputs.iter(), inputs.len(), &n => n * n),
///     vec![1, 4, 9, 16]);
/// # }
/// ```
#[macro_export]
macro_rules! par_map_collect {
    ($pool:expr, $iter:expr, $len:expr, $t:pat => $f:expr) => {{
        let len = $len;
        let mut results = Vec::with_capacity(len);
        unsafe { results.set_len(len) }
        $pool.scoped(|scope| {
            for ($t, r) in $iter.zip(results.iter_mut()) {
                scope.execute(move || *r = $f)
            }
        });
        results
    }};
}
