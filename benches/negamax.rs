#[macro_use]
extern crate bencher;
extern crate minimax;
#[path = "../examples/connect4.rs"]
mod connect4;

use bencher::Bencher;
use minimax::*;

fn bench_negamax(b: &mut Bencher) {
    let board = connect4::Board::default();
    b.iter(|| {
        let mut s = Negamax::<connect4::BasicEvaluator>::new(Options { max_depth: 5 });
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

fn bench_iterative(b: &mut Bencher) {
    let board = connect4::Board::default();
    b.iter(|| {
        let mut s = IterativeSearch::<connect4::BasicEvaluator>::new(
            IterativeOptions::default().with_table_byte_size(128_000).with_max_depth(5),
        );
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

benchmark_group!(benches, bench_negamax, bench_iterative);
benchmark_main!(benches);
