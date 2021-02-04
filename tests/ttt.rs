extern crate minimax;

#[path="../examples/ttt.rs"]
mod ttt;

use minimax::util::battle_royale;

// Ensure that two players using negamax always results in a draw.
#[test]
fn test_ttt_negamax_always_draws() {
    use minimax::strategies::negamax::{Negamax, Options};
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    for _ in 0..100 {
        assert_eq!(battle_royale(&mut s1, &mut s2), None);
    }
}

// Ensure that a player using negamax against a random one always results in
// either a draw or a win for the former player.
#[test]
fn test_ttt_negamax_vs_random_always_wins_or_draws() {
    use minimax::strategies::negamax::{Negamax, Options};
    use minimax::strategies::random::Random;
    let mut s1 = Negamax::<ttt::Evaluator>::new(Options { max_depth: 10 });
    let mut s2 = Random::new();
    for _ in 0..100 {
        assert_ne!(battle_royale(&mut s1, &mut s2), Some(1));
    }
}
