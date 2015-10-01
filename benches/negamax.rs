#![feature(test)]
extern crate minimax;
extern crate test;
use test::Bencher;
use minimax::*;

#[derive(Clone)]
pub struct Board;

#[derive(Copy, Clone)]
pub struct Place;

pub struct Eval;

pub struct Noop;

impl Move for Place {
    type G = Noop;
    fn apply(&self, _: &mut Board) { }
    fn undo(&self, _: &mut Board) { }
}

impl Game for Noop {
    type S = Board;
    type M = Place;

    fn generate_moves(_: &Board, _: Player, ms: &mut [Option<Place>]) -> usize {
        ms[0] = Some(Place);
        ms[1] = Some(Place);
        ms[2] = Some(Place);
        ms[3] = Some(Place);
        ms[4] = None;
        4
    }

    fn get_winner(_: &Board) -> Option<Winner> { None }
}

impl Evaluator for Eval {
    type G = Noop;

    fn evaluate(_: &Board, _: Option<Winner>) -> Evaluation {
        Evaluation::Score(0)
    }
}

#[bench]
fn bench_negamax(b: &mut Bencher) {
    let board = Board;
    let mut s = Negamax::<Eval>::new(Options { max_depth: 10 });
    b.iter(|| s.choose_move(&board, Player::Computer));
}
