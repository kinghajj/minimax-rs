extern crate minimax;

use crate::minimax::{Game, Strategy};
use std::fmt;

#[derive(Copy, Clone)]
struct Board {
    // First index by player.
    // Next index by pit, counting down from 6 to 1 for the pits in play.
    // Pit zero is that player's store.
    // If I wanted to be crazy bit twiddly I could put these in a pair of u64s and shift stuff around.
    pits: [[u8; 7]; 2],
    skipped: bool,
    // u1 of pits player index.
    to_move: bool,
}

impl Default for Board {
    fn default() -> Board {
        Board { pits: [[0, 4, 4, 4, 4, 4, 4]; 2], skipped: false, to_move: false }
    }
}

// 1-6 means play from that pit.
// 0 means pass (because of being skipped).
type Move = u8;

struct Mancala;

impl minimax::Game for Mancala {
    type S = Board;
    type M = Move;

    fn generate_moves(board: &Board, moves: &mut Vec<Move>) {
        if board.skipped {
            moves.push(0);
            return;
        }
        for i in 1..7 {
            if board.pits[board.to_move as usize][i] > 0 {
                moves.push(i as Move);
            }
        }
    }

    fn apply(board: &mut Board, m: Move) -> Option<Board> {
        let mut board = board.clone();
        if board.skipped {
            board.skipped = false;
            board.to_move = !board.to_move;
            return Some(board);
        }

        // Grab the stones.
        let mut player = board.to_move as usize;
        let mut i = m as usize;
        let mut stones = board.pits[player][i];
        board.pits[player][i] = 0;
        // At the beginning of each iteration, it points at the previous pit.
        while stones > 0 {
            if player == board.to_move as usize && i == 0 {
                i = 6;
                player ^= 1;
            } else if player != board.to_move as usize && i == 1 {
                i = 6;
                player ^= 1;
            } else {
                i -= 1;
            }
            board.pits[player][i] += 1;
            stones -= 1;
        }

        if player == board.to_move as usize {
            if i == 0 {
                // End condition: ends in own bowl
                board.skipped = true;
            } else if board.pits[player][i] == 1 {
                // End condition: ends on own side in empty pit
                let captured = board.pits[player][i] + board.pits[player ^ 1][7 - i];
                board.pits[player][i] = 0;
                board.pits[player ^ 1][7 - i] = 0;
                board.pits[player][0] += captured;
            }
        }

        board.to_move = !board.to_move;
        Some(board)
    }

    fn get_winner(board: &Board) -> Option<minimax::Winner> {
        if board.pits[0][1..].iter().sum::<u8>() == 0 || board.pits[1][1..].iter().sum::<u8>() == 0
        {
            let to_move_total = board.pits[board.to_move as usize].iter().sum::<u8>();
            Some(if to_move_total == 24 {
                minimax::Winner::Draw
            } else if to_move_total > 24 {
                minimax::Winner::PlayerToMove
            } else {
                minimax::Winner::PlayerJustMoved
            })
        } else {
            None
        }
    }

    fn zobrist_hash(board: &Board) -> u64 {
        let mut hash = board.to_move as u64;
        for i in 0..7 {
            hash ^= HASHES[i].wrapping_mul(board.pits[0][i] as u64);
            hash ^= HASHES[i + 7].wrapping_mul(board.pits[1][i] as u64);
        }
        hash
    }

    fn null_move(_: &Board) -> Option<Move> {
        Some(0)
    }

    fn notation(_: &Board, m: Move) -> Option<String> {
        Some(if m == 0 { "skipped".to_owned() } else { format!("pit {}", m) })
    }

    fn table_index(m: Move) -> u16 {
        m as u16
    }
    fn max_table_index() -> u16 {
        6
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "+-----------------------+\n|  |")?;
        for pit in &self.pits[1][1..] {
            write!(f, "{:>2}|", pit)?;
        }
        write!(f, "  |\n+{:>2}+--+--+--+--+--+--+{:>2}+\n|  ", self.pits[1][0], self.pits[0][0])?;
        for pit in self.pits[0][1..].iter().rev() {
            write!(f, "|{:>2}", pit)?;
        }
        write!(f, "|  |\n+-----------------------+\n")
    }
}

#[derive(Default)]
struct Evaluator;

impl minimax::Evaluator for Evaluator {
    type G = Mancala;
    fn evaluate(&self, board: &Board) -> minimax::Evaluation {
        board.pits[board.to_move as usize].iter().sum::<u8>() as minimax::Evaluation - 24
    }
}

fn main() {
    let mut board = Board::default();
    let opts = minimax::IterativeOptions::new().verbose();
    let mut strategy = minimax::IterativeSearch::new(Evaluator::default(), opts);
    strategy.set_timeout(std::time::Duration::from_secs(1));
    while Mancala::get_winner(&board).is_none() {
        println!("{}", board);
        match strategy.choose_move(&board) {
            Some(m) => board = Mancala::apply(&mut board, m).unwrap(),
            None => break,
        }
    }
    println!("Winner player {:?}", board.to_move as u8 + 1);
}

const HASHES: [u64; 14] = [
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
];
