#![no_std]

extern crate rand;

pub mod interface;
pub mod strategies;
pub mod util;

pub use interface::{Evaluation, Evaluator, Game, Move, Player, Strategy, Winner};
pub use strategies::negamax::{Negamax, Options};
