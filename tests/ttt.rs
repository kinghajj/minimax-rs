extern crate minimax;
extern crate num_cpus;
extern crate rand;

#[path="../examples/ttt.rs"]
mod ttt;

use minimax::util::battle_royale;

// Ensure that two players using negamax always results in a draw.
#[test]
fn test_ttt_negamax_always_draws() {
    use minimax::graders::negamax::{Negamax, ParallelNegamax, Options};
    use minimax::choosers::Random;
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut ps1 = ParallelNegamax::<ttt::Evaluator>::new(num_cpus::get() as u32,
                                                         Options { max_depth: 10 });
    let mut ps2 = ParallelNegamax::<ttt::Evaluator>::new(num_cpus::get() as u32,
                                                         Options { max_depth: 10 });
    let mut c1 = Random::new(rand::thread_rng());
    let mut c2 = Random::new(rand::thread_rng());
    for _ in 0..100 {
        assert!(battle_royale(&mut s1, &mut c1, &mut s2, &mut c2) == minimax::Winner::Draw);
        assert!(battle_royale(&mut ps1, &mut c1, &mut ps2, &mut c2) == minimax::Winner::Draw)
    }
}


// Ensure that a player using negamax against a random one always results in
// either a draw or a win for the former player.
#[test]
fn test_ttt_negamax_vs_random_always_wins_or_draws() {
    use minimax::graders::bogus::Bogus;
    use minimax::graders::negamax::{Negamax, ParallelNegamax, Options};
    use minimax::choosers::Random;
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Bogus;
    let mut ps1 = ParallelNegamax::<ttt::Evaluator>::new(num_cpus::get() as u32,
                                                         Options { max_depth: 10 });
    let mut c1 = Random::new(rand::thread_rng());
    let mut c2 = Random::new(rand::thread_rng());
    for _ in 0..100 {
        assert!(battle_royale(&mut s1, &mut c1, &mut s2, &mut c2) !=
                minimax::Winner::Competitor(minimax::Player::Opponent));
        assert!(battle_royale(&mut ps1, &mut c1, &mut s2, &mut c2) !=
                minimax::Winner::Competitor(minimax::Player::Opponent));

    }
}
