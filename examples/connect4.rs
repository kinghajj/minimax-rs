//! A definition of the game Connect Four using the library, for use in tests and benchmarks.
#![allow(dead_code)]

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
    all_pieces: u64,
    pub pieces_to_move: u64,
    num_moves: u8,
    hash: u64,
}

const NUM_COLS: u32 = 7;
const NUM_ROWS: u32 = 6;
const HEIGHT: u32 = NUM_ROWS + 1;
const COL_MASK: u64 = (1 << NUM_ROWS) - 1;

impl Board {
    fn reds_move(&self) -> bool {
        self.num_moves & 1 == 0
    }

    pub fn pieces_just_moved(&self) -> u64 {
        self.all_pieces ^ self.pieces_to_move
    }

    fn update_hash(&mut self, piece: u64) {
        // Lookup the hash for this position and this color.
        let position = piece.trailing_zeros() as usize;
        let color = self.num_moves as usize & 1;
        self.hash ^= HASHES[(position << 1) | color];
    }
}

impl Default for Board {
    fn default() -> Board {
        Board { all_pieces: 0, pieces_to_move: 0, num_moves: 0, hash: 0 }
    }
}

impl minimax::Zobrist for Board {
    fn zobrist_hash(&self) -> u64 {
        self.hash
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let red_pieces =
            if self.reds_move() { self.pieces_to_move } else { self.pieces_just_moved() };
        let yellow_pieces =
            if self.reds_move() { self.pieces_just_moved() } else { self.pieces_to_move };
        for row in (0..6).rev() {
            for col in 0..7 {
                write!(
                    f,
                    "{}",
                    if red_pieces >> (row + col * HEIGHT) & 1 != 0 {
                        '\u{1F534}'
                    } else if yellow_pieces >> (row + col * HEIGHT) & 1 != 0 {
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
        let col = (b.all_pieces >> self.col_shift()) & COL_MASK;
        let new_piece = (col + 1) << self.col_shift();
        // Swap colors
        b.pieces_to_move ^= b.all_pieces;
        b.all_pieces |= new_piece;
        b.num_moves += 1;
        b.update_hash(new_piece);
    }

    fn undo(&self, b: &mut Board) {
        let col = (b.all_pieces >> self.col_shift()) & COL_MASK;
        let prev_piece = (col ^ (col >> 1)) << self.col_shift();
        b.all_pieces &= !prev_piece;
        // Swap colors
        b.pieces_to_move ^= b.all_pieces;
        b.update_hash(prev_piece);
        b.num_moves -= 1;
    }
}

pub struct Game;

impl minimax::Game for Game {
    type S = Board;
    type M = Place;

    fn generate_moves(b: &Board, moves: &mut Vec<Place>) {
        let mut cols = b.all_pieces;
        for i in 0..NUM_COLS {
            if cols & COL_MASK < COL_MASK {
                moves.push(Place { col: i as u8 });
            }
            cols >>= HEIGHT;
        }
    }

    fn get_winner(b: &Board) -> Option<minimax::Winner> {
        // Position of pieces for the player that just moved.
        let pieces = b.pieces_just_moved();

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
    fn evaluate(&self, _: &Board) -> minimax::Evaluation {
        0
    }
}

impl Board {
    // Return bitmap of all open locations that would complete a four in a row for the given player.
    fn find_fourth_moves(&self, pieces: u64) -> u64 {
        let mut all = self.all_pieces;
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

#[derive(Clone)]
pub struct BasicEvaluator;

impl Default for BasicEvaluator {
    fn default() -> Self {
        Self {}
    }
}

impl minimax::Evaluator for BasicEvaluator {
    type G = Game;
    fn evaluate(&self, b: &Board) -> minimax::Evaluation {
        let player_pieces = b.pieces_to_move;
        let opponent_pieces = b.pieces_just_moved();
        let mut player_wins = b.find_fourth_moves(player_pieces);
        let mut opponent_wins = b.find_fourth_moves(opponent_pieces);

        let mut score = 0;
        // Bonus points for moves in the middle columns.
        for col in 2..5 {
            score +=
                ((player_pieces >> (HEIGHT * col)) & COL_MASK).count_ones() as minimax::Evaluation;
            score -= ((opponent_pieces >> (HEIGHT * col)) & COL_MASK).count_ones()
                as minimax::Evaluation;
        }

        // Count columns that cause immediate win.
        // Count columns that then allow immediate win.
        let mut all = b.all_pieces;
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

fn main() {
    use minimax::*;

    let mut b = Board::default();

    if std::env::args().any(|arg| arg == "perft") {
        perft::<self::Game>(&mut b, 10, false);
        return;
    }

    let mut dumb = IterativeSearch::new(
        BasicEvaluator::default(),
        IterativeOptions::new().with_double_step_increment(),
    );
    dumb.set_max_depth(8);

    let opts =
        IterativeOptions::new().with_table_byte_size(64_000_000).with_double_step_increment();
    let mut iterative =
        IterativeSearch::new(BasicEvaluator::default(), opts.clone().with_aspiration_window(5));
    iterative.set_max_depth(12);
    let mut parallel = ParallelSearch::new(BasicEvaluator::default(), opts, ParallelOptions::new());
    parallel.set_max_depth(12);

    let mut strategies: [&mut dyn Strategy<self::Game>; 3] =
        [&mut dumb, &mut iterative, &mut parallel];

    if std::env::args().any(|arg| arg == "parallel") {
        strategies.swap(1, 2);
    }

    let mut s = 0;
    while self::Game::get_winner(&b).is_none() {
        println!("{}", b);
        let ref mut strategy = strategies[s];
        match strategy.choose_move(&mut b) {
            Some(m) => {
                let color = if b.reds_move() { "Red" } else { "Yellow" };
                println!("{} piece in column {}", color, m.col + 1);
                m.apply(&mut b)
            }
            None => break,
        }
        s = 1 - s;
    }
    println!("{}", b);
}

// There aren't that many positions per color, so just encode the zobrist hash statically.
const HASHES: [u64; 100] = [
    0x73399349585d196e,
    0xe512dc15f0da3dd1,
    0x4fbc1b81c6197db2,
    0x16b5034810111a66,
    0xa9a9d0183e33c311,
    0xbb9d7bdea0dad2d6,
    0x089d9205c11ca5c7,
    0x18d9db91aa689617,
    0x1336123120681e34,
    0xc902e6c0bd6ef6bf,
    0x16985ba0916238c1,
    0x6144c3f2ab9f6dc4,
    0xf24b4842de919a02,
    0xdd6dd35ba0c150a1,
    0x369a9de8ec3676e3,
    0x2c366fb99be782d8,
    0x24d3231335c0dbd6,
    0x14048390c56e38f1,
    0x55dfbc820f635186,
    0x0dc98cb87372d5fa,
    0xe3098781582027b4,
    0x088158ec8202adca,
    0x231df62376ad9514,
    0xd3747fad069caeae,
    0x4e4f26cb41d0c620,
    0x06d0e37cd11b8f1c,
    0xed33865175fbbdd2,
    0xf1f52569481f0d8f,
    0xfb6fd5c922e2127c,
    0x6778bb0eba4a6649,
    0xe35b853bdac1210b,
    0x465a67712ec749a2,
    0x83b1fd78e576fe72,
    0xe84827644a5ccbe6,
    0x89095321ce8e4d03,
    0x298c529eecb0ec36,
    0xe9dcc93d77cb49ad,
    0xa7446daa1834c04a,
    0x93f15442b434d550,
    0x7f2a36dbf1cbce3f,
    0x03365a42023b02b3,
    0x101d87e850689cda,
    0x113b31e2760d2050,
    0x9cdb7b7394e1b0ae,
    0xd04530b3b7daf3a3,
    0x717e67aed6b4ffc9,
    0x4ae564a3f3ca8b03,
    0x07c50a4d89351437,
    0x7f3b32175e5f37e0,
    0x6e3599203bb50cd7,
    0xcfe2319d4a6cfa73,
    0xdbc6a398b10f5c3b,
    0x9c1ba28ae655bbd1,
    0x9dc87a426451941a,
    0x691e618354a55cb5,
    0x61b8cabbc575f4ba,
    0x7e6f31f1818593d4,
    0x9fa69e1ef4df8a9b,
    0x5a9dc96c3cb18d8f,
    0x65c4e9c0f40114f5,
    0x4e66504db2d937cf,
    0x4ebd6d097fe1e256,
    0xfb10983e639af6b1,
    0xcfbed7bd4032a59a,
    0x1f47f6a95049fe4f,
    0xbd461d202b879890,
    0xfc050073b0c74cbe,
    0x2923526a1f7092e9,
    0x0b1d30bb6b960bc7,
    0x632d12e4a9d0229d,
    0x8d4ffd6ab37c6bfd,
    0x561e36b8609b94ec,
    0x32e8482c9e7ed80c,
    0xaf62a119227b1029,
    0x62cb2a585410c311,
    0x7df3aeef90e1a0cb,
    0xe6d5a176f8a1b180,
    0x156e5162d8f2bef8,
    0xee84c58f5ebbe811,
    0xd32a1b4e24038bac,
    0xeaa1dbdbdd7731f7,
    0xedb554afd3d07cc6,
    0xbc789444317d4d05,
    0x0e23ce8f3d581fcd,
    0xacb498d4569249a8,
    0x843fb2519edc9f5a,
    0xe222f0eb79436809,
    0x7a88365f089ae80b,
    0x2a0f08694d7ea84d,
    0x09cad4dbfc990fa2,
    0xfe5f27499de6b4f8,
    0x3d8ed8ab1d44997f,
    0x2af64deca431f644,
    0xf2712b5274180c36,
    0x30eeae3a821bf86c,
    0x31c921831f06ad2f,
    0x40683ff11655cd2f,
    0xb78183a74cd6cb03,
    0xde9e15a6f99bda2f,
    0xa5293988641edb9b,
];
