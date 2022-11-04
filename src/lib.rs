extern crate rand;

pub mod interface;
pub mod strategies;
pub mod util;

pub use interface::{
    Evaluation, Evaluator, Game, Move, Strategy, Winner, Zobrist, BEST_EVAL, WORST_EVAL,
};
pub use strategies::iterative::{IterativeOptions, IterativeSearch, Replacement};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use strategies::mcts::{MCTSOptions, MonteCarloTreeSearch};
pub use strategies::negamax::Negamax;
pub use strategies::random::Random;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use strategies::ybw::{ParallelYbw, YbwOptions};
pub use util::perft;
