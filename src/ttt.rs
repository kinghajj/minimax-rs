//! A definition of the game Tic-Tac-Toe using the library, for use in tests.
//!
//! For example, playing a correctly-implemented strategy against itself should
//! always result in a draw; and playing such a strategy against one that picks
//! moves randomly should always result in a win or draw.

use interface;
use std::default::Default;
use std::fmt::{Display, Formatter, Result};
use std::convert::From;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Square {
    Empty,
    X,
    O,
}

impl Default for Square {
  fn default() -> Square { Square::Empty }
}

impl Display for Square {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "{}", match *self {
        Square::Empty => ' ',
        Square::X => 'X',
        Square::O => 'O',
    })
  }
}

impl From<interface::Player> for Square {
    fn from(p: interface::Player) -> Square {
        match p {
            interface::Player::Computer => Square::X,
            interface::Player::Opponent => Square::O,
        }
    }
}

impl From<Square> for interface::Player {
    fn from(s: Square) -> interface::Player {
        match s {
            Square::X => interface::Player::Computer,
            Square::O => interface::Player::Opponent,
            _ => panic!("From::from(Square::Empty))"),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    squares: [Square; 9]
}

impl Default for Board {
    fn default() -> Board {
        Board {
            squares: [Square::default(); 9]
        }
    }
}

impl Display for Board {
  fn fmt(&self, f: &mut Formatter) -> Result {
    try!(writeln!(f, "{} | {} | {}", self.squares[0], self.squares[1], self.squares[2]));
    try!(writeln!(f, "{} | {} | {}", self.squares[3], self.squares[4], self.squares[5]));
    try!(writeln!(f, "{} | {} | {}", self.squares[6], self.squares[7], self.squares[8]));
    Ok(())
  }
}

pub struct Game;

impl interface::Game for Game {
    type S = Board;
    type M = Place;

    fn generate_moves(b: &Board, p: interface::Player, ms: &mut [Option<Place>])
        -> usize {
        let mut j = 0;
        for i in 0..b.squares.len() {
            if b.squares[i] == Square::Empty {
                ms[j] = Some(Place { i: i as u8, s: From::from(p) });
                j += 1;
            }
        }
        ms[j] = None;
        j
    }

    fn get_winner(b: &Board) -> Option<interface::Winner> {
        // horizontal wins
        if b.squares[0] != Square::Empty && b.squares[0] == b.squares[1] && b.squares[1] == b.squares[2] {
            return Some(interface::Winner::Competitor(From::from(b.squares[0])));
        }
        if b.squares[3] != Square::Empty && b.squares[3] == b.squares[4] && b.squares[4] == b.squares[5] {
            return Some(interface::Winner::Competitor(From::from(b.squares[3])));
        }
        if b.squares[6] != Square::Empty && b.squares[6] == b.squares[7] && b.squares[7] == b.squares[8] {
            return Some(interface::Winner::Competitor(From::from(b.squares[6])));
        }
        // vertical wins
        if b.squares[0] != Square::Empty && b.squares[0] == b.squares[3] && b.squares[3] == b.squares[6] {
            return Some(interface::Winner::Competitor(From::from(b.squares[0])));
        }
        if b.squares[1] != Square::Empty && b.squares[1] == b.squares[4] && b.squares[4] == b.squares[7] {
            return Some(interface::Winner::Competitor(From::from(b.squares[1])));
        }
        if b.squares[2] != Square::Empty && b.squares[2] == b.squares[5] && b.squares[5] == b.squares[8] {
            return Some(interface::Winner::Competitor(From::from(b.squares[2])));
        }
        // diagonal wins
        if b.squares[0] != Square::Empty && b.squares[0] == b.squares[4] && b.squares[4] == b.squares[8] {
            return Some(interface::Winner::Competitor(From::from(b.squares[0])));
        }
        if b.squares[2] != Square::Empty && b.squares[2] == b.squares[4] && b.squares[4] == b.squares[6] {
            return Some(interface::Winner::Competitor(From::from(b.squares[2])));
        }
        // draws
        if b.squares.iter().all(|s| *s != Square::Empty) {
            Some(interface::Winner::Draw)
        } else {
            // non-terminal state
            None
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Place { i: u8, s: Square }

impl Display for Place {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}@{}", self.s, self.i)
    }
}

impl interface::Move for Place {
    type G = Game;
    fn apply(&self, b: &mut Board) {
        b.squares[self.i as usize] = self.s;
    }
    fn undo(&self, b: &mut Board) {
        b.squares[self.i as usize] = Square::Empty;
    }
}

pub struct Evaluator;

impl interface::Evaluator for Evaluator {
    type G = Game;
    // adapted from http://www.cs.olemiss.edu/~dwilkins/CSCI531/tic.c
    fn evaluate(b: &Board, mw: Option<interface::Winner>) -> interface::Evaluation {
        match mw {
            Some(interface::Winner::Competitor(wp)) => match wp {
                interface::Player::Computer => return interface::Evaluation::Best,
                interface::Player::Opponent => return interface::Evaluation::Worst,
            },
            _ => {},
        }
        let mut score = 0;

        // 3rd: check for doubles
        for i in 0..3 {
            let line = i * 3;
            if b.squares[line + 0] == b.squares[line + 1] {
                if b.squares[line + 0] == Square::X {
                    score += 5;
                } else if b.squares[line + 0] == Square::O {
                    score -= 5;
                }
            }
            if b.squares[line + 1] == b.squares[line + 2] {
                if b.squares[line + 1] == Square::X {
                    score += 5;
                } else if b.squares[line + 1] == Square::O {
                    score += 5;
                }
            }
            if b.squares[i] == b.squares[3 + i] {
                if b.squares[i] == Square::X {
                    score += 5;
                } else if b.squares[i] == Square::O {
                    score -= 5;
                }
            }
            if b.squares[3 + i] == b.squares[6 + i] {
                if b.squares[3 + i] == Square::X {
                    score += 5;
                } else if b.squares[3 + i] == Square::O {
                    score -= 5;
                }
            }
        }
        // 2nd: check for the middle square
        if b.squares[4] == Square::X {
            score += 5;
        }
        if b.squares[4] == Square::O {
            score -= 5;
        }
        interface::Evaluation::Score(score)
    }
}
