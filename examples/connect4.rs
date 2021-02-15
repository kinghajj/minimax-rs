//! A definition of the game Connect Four using the library, for use in tests and benchmarks.

extern crate minimax;

use std::default::Default;
use std::fmt::{Display, Formatter, Result};

#[derive(Clone)]
pub struct Board {
    // Some bitboard ideas from http://blog.gamesolver.org/solving-connect-four/06-bitboard/
    /* bit order example:
     * Leaves a blank row on top.
     *  5 12 19 26 33 40 47
     *  4 11 18 25 32 39 46
     *  3 10 17 24 31 38 45
     *  2  9 16 23 30 37 44
     *  1  8 15 22 29 36 43
     *  0  7 14 21 28 35 42
     */
    red_pieces: u64,
    yellow_pieces: u64,
    reds_move: bool,
    num_moves: u8,
}

const NUM_COLS: u32 = 7;
const NUM_ROWS: u32 = 6;
const HEIGHT: u32 = NUM_ROWS + 1;
const COL_MASK: u64 = (1 << NUM_ROWS) - 1;

impl Board {
    pub fn all_pieces(&self) -> u64 {
        self.red_pieces | self.yellow_pieces
    }

    pub fn pieces_to_move(&self) -> u64 {
        if self.reds_move {
            self.red_pieces
        } else {
            self.yellow_pieces
        }
    }
}

impl Default for Board {
    fn default() -> Board {
        Board { red_pieces: 0, yellow_pieces: 0, reds_move: true, num_moves: 0 }
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result {
        for row in (0..6).rev() {
            for col in 0..7 {
                write!(
                    f,
                    "{}",
                    if self.red_pieces >> (row + col * HEIGHT) & 1 != 0 {
                        '\u{1F534}'
                    } else if self.yellow_pieces >> (row + col * HEIGHT) & 1 != 0 {
                        '\u{1F7E1}'
                    } else {
                        '\u{25ef}'
                    }
                )?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
pub struct Place {
    col: u8,
}

impl Place {
    fn col_shift(&self) -> u32 {
        self.col as u32 * HEIGHT
    }
}

impl minimax::Move for Place {
    type G = Game;
    fn apply(&self, b: &mut Board) {
        let col = (b.all_pieces() >> self.col_shift()) & COL_MASK;
        let new_piece = (col + 1) << self.col_shift();
        if b.reds_move {
            b.red_pieces |= new_piece;
        } else {
            b.yellow_pieces |= new_piece;
        }
        b.reds_move = !b.reds_move;
        b.num_moves += 1;
    }

    fn undo(&self, b: &mut Board) {
        let col = (b.all_pieces() >> self.col_shift()) & COL_MASK;
        let prev_piece = (col ^ (col >> 1)) << self.col_shift();
        b.reds_move = !b.reds_move;
        if b.reds_move {
            b.red_pieces &= !prev_piece;
        } else {
            b.yellow_pieces &= !prev_piece;
        }
        b.num_moves -= 1;
    }
}

pub struct Game;

impl minimax::Game for Game {
    type S = Board;
    type M = Place;

    fn generate_moves(b: &Board, moves: &mut [Option<Place>]) -> usize {
        let mut n = 0;
        let mut cols = b.all_pieces();
        for i in 0..NUM_COLS {
            if cols & COL_MASK < COL_MASK {
                moves[n] = Some(Place { col: i as u8 });
                n += 1;
            }
            cols >>= HEIGHT;
        }
        moves[n] = None;
        n
    }

    fn get_winner(b: &Board) -> Option<minimax::Winner> {
        // Position of pieces for the player that just moved.
        let pieces = if b.reds_move { b.yellow_pieces } else { b.red_pieces };

        // Detect pairs of two pieces in a row, then pairs of two pairs in a
        // row.
        let matches = |shift| -> bool {
            let pairs = pieces & (pieces >> shift);
            pairs & (pairs >> 2 * shift) != 0
        };

        if matches(1) || matches(HEIGHT) || matches(HEIGHT + 1) || matches(HEIGHT - 1) {
            return Some(minimax::Winner::PlayerJustMoved);
        }

        // Full board with no winner.
        if b.num_moves as u32 == NUM_ROWS * NUM_COLS {
            Some(minimax::Winner::Draw)
        } else {
            None
        }
    }
}

pub struct DumbEvaluator;

impl minimax::Evaluator for DumbEvaluator {
    type G = Game;
    fn evaluate(_: &Board) -> minimax::Evaluation {
        0
    }
}

impl Board {
    // Return bitmap of all open locations that would complete a four in a row for the given player.
    fn find_fourth_moves(&self, pieces: u64) -> u64 {
        let mut all = self.all_pieces();
        // Mark the fake row on top as full to prevent wrapping around.
        let mut top_row = COL_MASK + 1;
        for _ in 0..NUM_COLS {
            all |= top_row;
            top_row <<= HEIGHT;
        }

        let matches = |shift| -> u64 {
            let pairs = pieces & (pieces >> shift); // Pairs of this color.
            let singles = (pieces >> shift) & !all | (pieces << shift) & !all; // One of this color and one empty.
            (pairs >> shift * 2) & singles | (pairs << shift * 2) & singles
        };

        // Vertical
        matches(1) |
	// Horizontal
	matches(HEIGHT) |
	// Diagonal
	matches(HEIGHT+1) |
	// Other diagonal
	matches(HEIGHT-1)
    }
}

pub struct BasicEvaluator;

impl minimax::Evaluator for BasicEvaluator {
    type G = Game;
    fn evaluate(b: &Board) -> minimax::Evaluation {
        let player_pieces = if b.reds_move { b.red_pieces } else { b.yellow_pieces };
        let opponent_pieces = if b.reds_move { b.yellow_pieces } else { b.red_pieces };
        let mut player_wins = b.find_fourth_moves(player_pieces);
        let mut opponent_wins = b.find_fourth_moves(opponent_pieces);

        let mut score = 0;
        // Bonus points for moves in the middle columns.
        for col in 2..5 {
            score += ((player_pieces >> (HEIGHT * col)) & COL_MASK).count_ones() as i32;
            score -= ((opponent_pieces >> (HEIGHT * col)) & COL_MASK).count_ones() as i32;
        }

        // Count columns that cause immediate win.
        // Count columns that then allow immediate win.
        let mut all = b.all_pieces();
        for _ in 0..NUM_COLS {
            let next_move = (all & COL_MASK) + 1;
            if next_move > COL_MASK {
                continue;
            }
            if next_move & player_wins != 0 {
                score += 10;
            }
            if next_move & opponent_wins != 0 {
                score -= 10;
            }
            let afterwards_move = next_move << 1;
            if afterwards_move & player_wins != 0 {
                score += 5;
            }
            if afterwards_move & opponent_wins != 0 {
                score -= 5;
            }

            all >>= HEIGHT;
            player_wins >>= HEIGHT;
            opponent_wins >>= HEIGHT;
        }

        score
    }
}

#[cfg(not(test))]
fn main() {
    use minimax::strategies::negamax::{Negamax, Options};
    use minimax::{Game, Move, Strategy};

    let mut b = Board::default();
    let mut strategies: [&mut dyn Strategy<self::Game>; 2] = [
        &mut Negamax::<DumbEvaluator>::new(Options { max_depth: 8 }),
        &mut Negamax::<BasicEvaluator>::new(Options { max_depth: 8 }),
    ];
    let mut s = 0;
    while self::Game::get_winner(&b).is_none() {
        println!("{}", b);
        let ref mut strategy = strategies[s];
        match strategy.choose_move(&mut b) {
            Some(m) => {
                let color = if b.reds_move { "Red" } else { "Yellow" };
                println!("{} piece in column {}", color, m.col + 1);
                m.apply(&mut b)
            }
            None => break,
        }
        s = 1 - s;
    }
    println!("{}", b);
}
