extern crate rand;

pub mod interface;
pub mod strategies;
pub mod util;

pub use interface::{Evaluation, Evaluator, Game, Move, Strategy, Winner, Zobrist};
pub use strategies::iterative::{IterativeOptions, IterativeSearch};
pub use strategies::negamax::Negamax;
