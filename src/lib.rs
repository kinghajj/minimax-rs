extern crate rand;

pub mod interface;
pub mod strategies;
pub mod util;

pub use interface::{
    Evaluation, Evaluator, Game, Move, Strategy, Winner, Zobrist, BEST_EVAL, WORST_EVAL,
};
pub use strategies::iterative::{IterativeOptions, IterativeSearch, Replacement};
pub use strategies::lazy_smp::{LazySmp, LazySmpOptions};
pub use strategies::mcts::{MCTSOptions, MonteCarloTreeSearch};
pub use strategies::negamax::Negamax;
pub use strategies::random::Random;
pub use strategies::ybw::{ParallelYbw, YbwOptions};
pub use util::perft;
