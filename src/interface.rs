//! The common structures and traits.

use std::ops;

/// An assessment of a game state from the perspective of the player about to move.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evaluation {
    /// An absolutely disastrous outcome, e.g. a loss.
    Worst,
    /// An outcome with some score. Higher values mean a more favorable state.
    /// A draw is defined as a score of zero.
    Score(i64),
    /// An absolutely wonderful outcome, e.g. a win.
    Best,
}

/// Negating an evaluation results in the corresponding one from the other
/// player's persective.
impl ops::Neg for Evaluation {
    type Output = Evaluation;
    #[inline]
    fn neg(self) -> Evaluation {
        match self {
            Evaluation::Worst => Evaluation::Best,
            Evaluation::Score(s) => Evaluation::Score(-s),
            Evaluation::Best => Evaluation::Worst,
        }
    }
}

/// Evaluates a game's positions.
pub trait Evaluator {
    /// The type of game that can be evaluated.
    type G: Game;
    /// Evaluate the non-terminal state from the persective of the player to
    /// move next.
    fn evaluate(s: &<Self::G as Game>::S) -> Evaluation;
}

/// Defines how a move affects the game state.
///
/// A move is able to change initial `Game` state, as well as revert the state.
/// This allows the game tree to be searched with a constant amount of space.
pub trait Move {
    /// The type of game that the move affects.
    type G: Game;
    /// Change the state of `S` so that the move is applied.
    fn apply(&self, &mut <Self::G as Game>::S);
    /// Revert the state of `S` so that the move is undone.
    fn undo(&self, &mut <Self::G as Game>::S);
}

/// The result of playing a game until it finishes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Winner {
    /// The player who made the last move won.
    PlayerJustMoved,
    /// Nobody won.
    Draw,
    /// The player who made the last move lost.
    ///
    /// This is uncommon, and many games (chess, checkers, tic-tac-toe, etc)
    /// do not have this possibility.
    PlayerToMove,
}

impl Winner {
    /// Canonical evaluations for end states.
    pub fn evaluate(&self) -> Evaluation {
	match *self {
	    Winner::PlayerJustMoved => Evaluation::Worst,
	    Winner::PlayerToMove => Evaluation::Best,
	    Winner::Draw => Evaluation::Score(0),
	}
    }
}

/// Defines the rules for a two-player, perfect-knowledge game.
///
/// A game ties together types for the state and moves, generates the possible
/// moves from a particular state, and determines whether a state is terminal.
pub trait Game : Sized {
    /// The type of the game state.
    type S;
    /// The type of game moves.
    type M: Move<G=Self>;

    /// Generate moves at the given state. After finishing, the next entry in
    /// the slice should be set to `None` to indicate the end.  Returns the
    /// number of moves generated.
    ///
    /// Currently, there's a deficiency that all strategies assume that at most
    /// 100 moves may be generated for any position, which allows the underlying
    /// memory for the slice to be a stack-allocated array. Once stable, this
    /// trait will be extended with an associated constant to specify the
    /// maximum number of moves.
    fn generate_moves(&Self::S, &mut [Option<Self::M>]) -> usize;

    /// Returns `Some(PlayerJustMoved)` or `Some(PlayerToMove)` if there's a winner,
    /// `Some(Draw)` if the state is terminal without a winner, and `None` if
    /// the state is non-terminal.
    fn get_winner(&Self::S) -> Option<Winner>;
}

/// Defines a method of choosing a move for the current player.
pub trait Strategy<G: Game> {
    fn choose_move(&mut self, &G::S) -> Option<G::M>;
}
