//! The `minimax` library provides interfaces for defining two-player
//! perfect-knowledge games, and strategies for choosing moves.
//!
//! Any game can be defined by implementing 2 traits: Game and Move.
//! ```
//! use minimax::Strategy;
//!
//! // Stateless rules object.
//! struct TugOfWar;
//! // State of the game.
//! #[derive(Clone)]
//! struct War(i8);
//! // A move that a player can make.
//! #[derive(Copy, Clone, Debug, Eq, PartialEq)]
//! struct Tug(i8);
//!
//! impl minimax::Game for TugOfWar {
//!     type S = War;
//!     type M = Tug;
//!
//!     fn generate_moves(s: &War, moves: &mut Vec<Tug>) {
//!         moves.push(Tug(-1));
//!         moves.push(Tug(1));
//!     }
//!
//!     fn get_winner(state: &War) -> Option<minimax::Winner> {
//!         if state.0 > 9 {
//!             Some(if state.0 % 2 == 0 {
//!                 minimax::Winner::PlayerJustMoved
//!             } else {
//!                 minimax::Winner::PlayerToMove
//!             })
//!         } else if state.0 < 9 {
//!             Some(if state.0 % 2 == 0 {
//!                 minimax::Winner::PlayerToMove
//!             } else {
//!                 minimax::Winner::PlayerJustMoved
//!             })
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! impl minimax::Move for Tug {
//!     type G = TugOfWar;
//!     fn apply(&self, state: &mut War) {
//!         state.0 += self.0
//!     }
//!     fn undo(&self, state: &mut War) {
//!         state.0 -= self.0
//!     }
//! }
//!
//! // To run the search we need an evaluator.
//! struct Eval;
//! impl minimax::Evaluator for Eval {
//!     type G = TugOfWar;
//!     fn evaluate(&self, state: &War) -> minimax::Evaluation {
//!         if state.0 % 2 == 0 {
//!             state.0 as minimax::Evaluation
//!         } else {
//!             -state.0 as minimax::Evaluation
//!         }
//!     }
//! }
//!
//! // Now we can use a simple Strategy to find a move from the initial state.
//! let start = War(0);
//! let mut strategy = minimax::Negamax::new(Eval{}, 3);
//! let best_move = strategy.choose_move(&start).unwrap();
//! ```

pub mod interface;
pub mod strategies;
pub mod util;

pub use interface::{
    Evaluation, Evaluator, Game, Move, Strategy, Winner, Zobrist, BEST_EVAL, WORST_EVAL,
};
pub use strategies::iterative::{IterativeOptions, IterativeSearch, Replacement};
#[cfg(not(target_arch = "wasm32"))]
pub use strategies::mcts::{MCTSOptions, MonteCarloTreeSearch};
pub use strategies::negamax::Negamax;
pub use strategies::random::Random;
#[cfg(not(target_arch = "wasm32"))]
pub use strategies::ybw::{ParallelOptions, ParallelSearch};
pub use util::perft;
