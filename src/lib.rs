extern crate rand;
extern crate scoped_threadpool;

#[macro_use]
pub mod util;

pub mod choosers;
pub mod graders;
pub mod interface;

pub use interface::{Chooser, Evaluation, Evaluator, Game, Grader, Move, Player, Winner};
pub use graders::negamax::{Negamax, ParallelNegamax, Options};
pub use choosers::{Deterministic, Random};
