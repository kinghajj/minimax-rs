extern crate chess;
extern crate minimax;

use chess::{Board, BoardStatus, ChessMove, MoveGen};
use minimax::{Game, Strategy};

struct Chess;

impl minimax::Game for Chess {
    type S = Board;
    type M = ChessMove;

    fn generate_moves(b: &Board, moves: &mut Vec<ChessMove>) {
        for m in MoveGen::new_legal(b) {
            moves.push(m);
        }
    }

    fn get_winner(b: &Board) -> Option<minimax::Winner> {
        match b.status() {
            BoardStatus::Ongoing => None,
            BoardStatus::Stalemate => Some(minimax::Winner::Draw),
            BoardStatus::Checkmate => Some(minimax::Winner::PlayerJustMoved),
        }
    }

    fn apply(b: &mut Board, m: ChessMove) -> Option<Board> {
        Some(b.make_move_new(m))
    }

    fn zobrist_hash(b: &Board) -> u64 {
        b.get_hash()
    }

    fn notation(_b: &Board, m: &ChessMove) -> Option<String> {
        Some(format!("{}", m))
    }
}

#[derive(Default)]
struct Evaluator;

impl minimax::Evaluator for Evaluator {
    type G = Chess;
    fn evaluate(&self, board: &Board) -> minimax::Evaluation {
        let mut score = 0;
        for sq in 0..64 {
            let sq = unsafe { chess::Square::new(sq) };
            if let Some(piece) = board.piece_on(sq) {
                let value = match piece {
                    chess::Piece::Pawn => 1,
                    chess::Piece::Knight => 3,
                    chess::Piece::Bishop => 3,
                    chess::Piece::Rook => 5,
                    chess::Piece::Queen => 9,
                    chess::Piece::King => 0,
                };
                if board.color_on(sq).unwrap() == board.side_to_move() {
                    score += value;
                } else {
                    score -= value;
                }
            }
        }
        score
    }
}

fn main() {
    let mut b = Board::default();
    let opts = minimax::IterativeOptions::new().verbose();
    let mut strategy = minimax::IterativeSearch::new(Evaluator::default(), opts);
    strategy.set_timeout(std::time::Duration::from_secs(1));
    while Chess::get_winner(&b).is_none() {
        println!("{}", b);
        match strategy.choose_move(&b) {
            Some(m) => b = Chess::apply(&mut b, m).unwrap(),
            None => break,
        }
    }
    println!("Checkmate {:?}", b.side_to_move());
}
