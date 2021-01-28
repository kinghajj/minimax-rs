//! The common structures and traits.

use core::ops;

/// A competitor within a game.
///
/// For simplicity, only two players are supported. Their values correspond to
/// the "color" parameter in Negamax.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i8)]
pub enum Player {
    Computer = 1,
    Opponent = -1,
}

/// Negating a player results in the opposite one.
impl ops::Neg for Player {
    type Output = Player;
    #[inline]
    fn neg(self) -> Player {
        match self {
            Player::Computer => Player::Opponent,
            Player::Opponent => Player::Computer,
        }
    }
}

/// An assessment of a game state from a particular player's perspective.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evaluation {
    /// An absolutely disastrous outcome, e.g. a loss.
    Worst,
    /// An outcome with some score. Higher values mean a more favorable state.
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

/// Multiplying a player and an evaluation negates the latter iff the former
/// is `Opponent`.
impl ops::Mul<Evaluation> for Player {
    type Output = Evaluation;
    #[inline]
    fn mul(self, e: Evaluation) -> Evaluation {
        match self {
            Player::Computer => e,
            Player::Opponent => -e,
        }
    }
}

/// Evaluates a game's positions.
///
/// The methods are defined recursively, so that implementing one is sufficient.
pub trait Evaluator {
    /// The type of game that can be evaluated.
    type G: Game;
    /// Evaluate the state from the persective of `Player::Computer`.
    #[inline]
    fn evaluate(s: &<Self::G as Game>::S, mw: Option<Winner>) -> Evaluation {
        Self::evaluate_for(s, mw, Player::Computer)
    }

    /// Evaluate the state from the given player's persective.
    #[inline]
    fn evaluate_for(s: &<Self::G as Game>::S, mw: Option<Winner>, p: Player) -> Evaluation {
        p * Self::evaluate(s, mw)
    }
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
    /// A player won.
    Competitor(Player),
    /// Nobody won.
    Draw,
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

    /// Generate moves for a player at the given state. After finishing, the
    /// next entry in the slice should be set to `None` to indicate the end.
    /// Returns the number of moves generated.
    ///
    /// Currently, there's a deficiency that all strategies assume that at most
    /// 100 moves may be generated for any position, which allows the underlying
    /// memory for the slice to be a stack-allocated array. One stable, this
    /// trait will be extended with an associated constant to specify the
    /// maximum number of moves.
    fn generate_moves(&Self::S, Player, &mut [Option<Self::M>]) -> usize;

    /// Returns `Some(Competitor(winning_player))` if there's a winner,
    /// `Some(Draw)` if the state is terminal without a winner, and `None` if
    /// the state is non-terminal.
    fn get_winner(&Self::S) -> Option<Winner>;
}

/// Defines a method of choosing a move for either player in a any game.
pub trait Strategy<G: Game> {
    fn choose_move(&mut self, &G::S, Player) -> Option<G::M>;
}
