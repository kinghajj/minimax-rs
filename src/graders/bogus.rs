//! A "bogus" grader that assigns the same value to every move.

use super::super::interface::*;

/// A grader that always evaluates every move to `Evaluation::Worst`.
pub struct Bogus;

impl<G: Game> Grader<G> for Bogus
    where G::M: Copy {
    fn grade(&mut self, s: &G::S, p: Player) -> Vec<Grade<G::M>> {
        let mut moves = [None; 100];
        let num_moves = G::generate_moves(s, p, &mut moves);
        moves.into_iter()
             .take(num_moves)
             .map(|m| m.unwrap())
             .map(|m| {
                 Grade {
                     value: Evaluation::Worst,
                     play: m,
                 }
             })
             .collect()
    }
}
