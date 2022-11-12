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
        let mut s = Negamax::new(connect4::BasicEvaluator::default(), 5);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

fn bench_iterative(b: &mut Bencher) {
    let board = connect4::Board::default();
    b.iter(|| {
        let mut s = IterativeSearch::new(
            connect4::BasicEvaluator::default(),
            IterativeOptions::new().with_table_byte_size(32_000),
        );
        s.set_max_depth(5);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

fn bench_parallel(b: &mut Bencher) {
    let board = connect4::Board::default();
    b.iter(|| {
        let mut s = ParallelSearch::new(
            connect4::BasicEvaluator::default(),
            IterativeOptions::new().with_table_byte_size(32_000),
            ParallelOptions::new(),
        );
        s.set_max_depth(5);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

benchmark_group!(benches, bench_negamax, bench_iterative, bench_parallel);
benchmark_main!(benches);
