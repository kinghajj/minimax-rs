extern crate rand;

pub mod interface;
pub mod strategies;
pub mod test;

pub mod ttt;

pub use interface::{Evaluation, Evaluator, Game, Move, Player, Strategy, Winner};
pub use strategies::negamax::{Negamax, Options};
