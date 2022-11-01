extern crate chess;
extern crate minimax;

use minimax::{Game, Move, Strategy};

struct Chess;

// Using newtypes to get external chess impl to implement minimax traits.
#[derive(Clone)]
struct Board {
    history: Vec<chess::Board>,
}
#[derive(Copy, Clone, Eq, PartialEq)]
struct ChessMove(chess::ChessMove);

impl Board {
    fn new() -> Self {
        Self { history: vec![chess::Board::default()] }
    }
    fn board(&self) -> &chess::Board {
        self.history.last().unwrap()
    }
}

impl minimax::Zobrist for Board {
    fn zobrist_hash(&self) -> u64 {
        self.board().get_hash()
    }
}

impl minimax::Game for Chess {
    type S = Board;
    type M = ChessMove;

    fn generate_moves(b: &Board, moves: &mut Vec<ChessMove>) {
        for m in chess::MoveGen::new_legal(b.board()) {
            moves.push(ChessMove(m));
        }
    }

    fn get_winner(b: &Board) -> Option<minimax::Winner> {
        match b.board().status() {
            chess::BoardStatus::Ongoing => None,
            chess::BoardStatus::Stalemate => Some(minimax::Winner::Draw),
            chess::BoardStatus::Checkmate => Some(minimax::Winner::PlayerJustMoved),
        }
    }
}

impl minimax::Move for ChessMove {
    type G = Chess;
    fn apply(&self, b: &mut Board) {
        b.history.push(b.board().make_move_new(self.0));
    }

    fn undo(&self, b: &mut Board) {
        b.history.pop();
    }

    fn notation(&self, _b: &Board) -> Option<String> {
        Some(format!("{}", self.0))
    }
}

#[derive(Default)]
struct Evaluator;

impl minimax::Evaluator for Evaluator {
    type G = Chess;
    fn evaluate(&self, b: &Board) -> minimax::Evaluation {
        let board = b.board();
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
    let mut b = Board::new();
    let opts = minimax::IterativeOptions::new().verbose();
    let mut strategy = minimax::IterativeSearch::new(Evaluator::default(), opts);
    strategy.set_timeout(std::time::Duration::from_secs(1));
    while Chess::get_winner(&b).is_none() {
        println!("{}", b.board());
        match strategy.choose_move(&b) {
            Some(m) => m.apply(&mut b),
            None => break,
        }
    }
    println!("Checkmate {:?}", b.board().side_to_move());
}
