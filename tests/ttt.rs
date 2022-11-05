extern crate minimax;

#[path = "../examples/ttt.rs"]
mod ttt;

use minimax::util::battle_royale;
use minimax::{MCTSOptions, MonteCarloTreeSearch, Negamax, Random};

// Ensure that two players using negamax always results in a draw.
#[test]
fn test_ttt_negamax_always_draws() {
    let mut s1 = Negamax::new(ttt::Evaluator::default(), 10);
    let mut s2 = Negamax::new(ttt::Evaluator::default(), 10);
    for _ in 0..100 {
        assert_eq!(battle_royale(&mut s1, &mut s2), None);
    }
}

// Ensure that a player using negamax against a random one always results in
// either a draw or a win for the former player.
#[test]
fn test_ttt_negamax_vs_random_always_wins_or_draws() {
    let mut s1 = Negamax::new(ttt::Evaluator::default(), 10);
    let mut s2 = Random::new();
    for _ in 0..100 {
        assert_ne!(battle_royale(&mut s1, &mut s2), Some(1));
    }
}

#[test]
fn test_ttt_mcts_vs_random_always_wins_or_draws() {
    let mut s1 = MonteCarloTreeSearch::new(MCTSOptions::default().with_num_threads(1));
    s1.set_max_rollouts(500);
    let mut s2 = Random::new();
    for _ in 0..10 {
        assert_ne!(battle_royale::<ttt::Game, _, _>(&mut s1, &mut s2), Some(1));
    }
}
