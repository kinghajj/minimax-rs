use super::super::interface::*;
use super::util::AtomicBox;

use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

struct Node<M> {
    // The Move to get from the parent to here.
    // Only None at the root.
    m: Option<M>,
    visits: AtomicU32,
    // +1 for wins, -1 for losses, +0 for draws.
    // From perspective of player to move.
    score: AtomicI32,
    // Lazily populated.
    expansion: AtomicBox<NodeExpansion<M>>,
}

struct NodeExpansion<M> {
    // Populated if this node is an end state.
    winner: Option<Winner>,
    children: Vec<Node<M>>,
}

fn new_expansion<G: Game>(state: &G::S) -> Box<NodeExpansion<G::M>> {
    let winner = G::get_winner(state);
    let children = if winner.is_some() {
        Vec::new()
    } else {
        let mut moves = Vec::new();
        G::generate_moves(state, &mut moves);
        moves.into_iter().map(|m| Node::new(Some(m))).collect::<Vec<_>>()
    };
    Box::new(NodeExpansion { winner, children })
}

impl<M> Node<M> {
    fn new(m: Option<M>) -> Self {
        Node {
            m,
            expansion: AtomicBox::default(),
            visits: AtomicU32::new(0),
            score: AtomicI32::new(0),
        }
    }

    // Choose best child based on UCT.
    fn best_child(&self, exploration_score: f32) -> Option<&Node<M>> {
        let log_visits = (self.visits.load(Ordering::SeqCst) as f32).log2();
        let expansion = self.expansion.get()?;
        // Find a node, randomly chosen among the best scores.
        // TODO: make it more uniformly random?
        let n = expansion.children.len();
        let mut i = rand::thread_rng().gen_range(0..n);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_child = None;
        for _ in 0..n {
            let score = expansion.children[i].uct_score(exploration_score, log_visits);
            debug_assert!(!score.is_nan());
            if score > best_score {
                best_score = score;
                best_child = Some(&expansion.children[i]);
            }
            i = (i + 1) % n;
        }
        best_child
    }

    fn uct_score(&self, exploration_score: f32, log_parent_visits: f32) -> f32 {
        let visits = self.visits.load(Ordering::Relaxed) as f32;
        let score = self.score.load(Ordering::Relaxed) as f32;
        if visits == 0.0 {
            // Avoid NaNs.
            return if exploration_score > 0.0 { f32::INFINITY } else { 0.0 };
        }
        let win_ratio = (score + visits) / (2.0 * visits);
        win_ratio + exploration_score * (2.0 * log_parent_visits / visits).sqrt()
    }
}

pub struct MonteCarloTreeSearch {
    // TODO: Evaluator

    // Config
    max_rollout_depth: usize,
    max_rollouts: u32,
    //max_time: Duration,
    // TODO: rollouts_per_node
    // TODO: num_threads
}

impl MonteCarloTreeSearch {
    pub fn new() -> Self {
        Self { max_rollout_depth: 200, max_rollouts: 100 }
    }

    // Returns score for this node. +1 for win of original player to move.
    // TODO: policy options: random, look 1 ahead for winning moves, BYO Evaluator.
    fn rollout<G: Game>(&self, s: &G::S) -> i32
    where
        G::S: Clone,
    {
        let mut rng = rand::thread_rng();
        let mut depth = self.max_rollout_depth;
        let mut state = s.clone();
        let mut moves = Vec::new();
        let mut sign = 1;
        loop {
            if let Some(winner) = G::get_winner(&state) {
                return match winner {
                    Winner::PlayerJustMoved => 1,
                    Winner::PlayerToMove => -1,
                    Winner::Draw => 0,
                } * sign;
            }

            if depth == 0 {
                return 0;
            }

            moves.clear();
            G::generate_moves(s, &mut moves);
            let m = moves.choose(&mut rng).unwrap();
            m.apply(&mut state);
            sign = -sign;
            depth -= 1;
        }
    }

    // Explore the tree, make a new node, rollout, backpropagate.
    fn simulate<G: Game>(&self, node: &Node<G::M>, state: &mut G::S, mut force_rollout: bool) -> i32
    where
        G::S: Clone,
    {
        if force_rollout {
            let result = self.rollout::<G>(state);

            // Backpropagate.
            node.visits.fetch_add(1, Ordering::SeqCst);
            node.score.fetch_add(result, Ordering::SeqCst);
            return result;
        }

        let expansion = node.expansion.get().unwrap_or_else(|| {
            // Expand this node, and force a rollout when we recurse.
            force_rollout = true;
            node.expansion.try_set(new_expansion::<G>(state))
        });

        if let Some(winner) = expansion.winner {
            let result = match winner {
                Winner::PlayerJustMoved => 1,
                Winner::PlayerToMove => -1,
                Winner::Draw => 0,
            };

            // Backpropagate.
            node.visits.fetch_add(1, Ordering::SeqCst);
            node.score.fetch_add(result, Ordering::SeqCst);
            return result;
        }

        // Recurse.
        let next = node.best_child(1.).unwrap();
        let m = next.m.as_ref().unwrap();
        m.apply(state);
        let result = -self.simulate::<G>(next, state, force_rollout);
        m.undo(state);

        // Backpropagate.
        node.visits.fetch_add(1, Ordering::SeqCst);
        node.score.fetch_add(result, Ordering::SeqCst);
        result
    }
}

impl<G: Game> Strategy<G> for MonteCarloTreeSearch
where
    G::S: Clone,
    G::M: Copy,
{
    fn choose_move(&mut self, s: &G::S) -> Option<G::M> {
        let root = Node::<G::M>::new(None);
        root.expansion.try_set(new_expansion::<G>(s));
        let mut state = s.clone();
        for _ in 0..self.max_rollouts {
            self.simulate::<G>(&root, &mut state, false);
        }
        debug_assert_eq!(self.max_rollouts, root.visits.load(Ordering::SeqCst));
        let exploration = 0.0; // Just get best node.
        root.best_child(exploration).map(|node| node.m.unwrap())
    }
}

mod tests {
    // TODO: make a fake game with branching_factor=1 to test correct signage of results.
    // TODO: make a game with branching_factor=2: add or subtract to shared total

    // or maybe just run tic tac toe against random many times and check that it always wins
}
