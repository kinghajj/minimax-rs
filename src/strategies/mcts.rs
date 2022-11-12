use super::super::interface::*;
use super::sync_util::*;

use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::spawn;
use std::time::Duration;

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
        let mut log_visits = (self.visits.load(Ordering::SeqCst) as f32).log2();
        // Keep this numerator non-negative.
        if log_visits < 0.0 {
            log_visits = 0.0;
        }

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

    fn update_stats(&self, result: i32) -> Option<i32> {
        self.visits.fetch_add(1, Ordering::SeqCst);
        self.score.fetch_add(result, Ordering::SeqCst);
        // Always return Some, as we aren't timed out.
        Some(result)
    }
}

/// Options for MonteCarloTreeSearch.
#[derive(Clone)]
pub struct MCTSOptions {
    max_rollout_depth: u32,
    rollouts_before_expanding: u32,
    // None means use num_cpus.
    num_threads: Option<usize>,
    // TODO: rollout_policy
}

impl Default for MCTSOptions {
    fn default() -> Self {
        Self { max_rollout_depth: 100, rollouts_before_expanding: 0, num_threads: None }
    }
}

impl MCTSOptions {
    /// Set a maximum depth for rollouts. Rollouts that reach this depth are
    /// stopped and assigned a Draw value.
    pub fn with_max_rollout_depth(mut self, depth: u32) -> Self {
        self.max_rollout_depth = depth;
        self
    }

    /// How many rollouts to run on a single leaf node before expanding its
    /// children. The default value is 0, where every rollout expands some
    /// leaf node.
    pub fn with_rollouts_before_expanding(mut self, rollouts: u32) -> Self {
        self.rollouts_before_expanding = rollouts;
        self
    }

    /// How many threads to run. Defaults to num_cpus.
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }
}

/// A strategy that uses random playouts to explore the game tree to decide on the best move.
/// This can be used without an Evaluator, just using the rules of the game.
pub struct MonteCarloTreeSearch<G: Game> {
    // TODO: Evaluator
    options: MCTSOptions,
    max_rollouts: u32,
    max_time: Duration,
    timeout: Arc<AtomicBool>,
    game_type: PhantomData<G>,
}

// derive is broken with PhantomData (https://github.com/rust-lang/rust/issues/26925)
impl<G: Game> Clone for MonteCarloTreeSearch<G> {
    fn clone(&self) -> Self {
        Self {
            options: self.options.clone(),
            max_rollouts: self.max_rollouts,
            max_time: self.max_time,
            timeout: self.timeout.clone(),
            game_type: PhantomData,
        }
    }
}

impl<G: Game> MonteCarloTreeSearch<G> {
    pub fn new(options: MCTSOptions) -> Self {
        Self {
            options,
            max_rollouts: 0,
            max_time: Duration::from_secs(5),
            timeout: Arc::new(AtomicBool::new(false)),
            game_type: PhantomData,
        }
    }

    /// Instead of a timeout, run this many rollouts to choose a move.
    pub fn set_max_rollouts(&mut self, rollouts: u32) {
        self.max_time = Duration::default();
        self.max_rollouts = rollouts;
    }

    // Returns score for this node. +1 for win of original player to move.
    // TODO: policy options: random, look 1 ahead for winning moves, BYO Evaluator.
    fn rollout(&self, s: &G::S) -> i32
    where
        G::S: Clone,
    {
        let mut rng = rand::thread_rng();
        let mut depth = self.options.max_rollout_depth;
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
            G::generate_moves(&state, &mut moves);
            let m = moves.choose(&mut rng).unwrap();
            m.apply(&mut state);
            sign = -sign;
            depth -= 1;
        }
    }

    // Explore the tree, make a new node, rollout, backpropagate.
    fn simulate(&self, node: &Node<G::M>, state: &mut G::S, mut force_rollout: bool) -> Option<i32>
    where
        G::S: Clone,
    {
        if self.timeout.load(Ordering::Relaxed) {
            return None;
        }
        if force_rollout {
            return node.update_stats(self.rollout(state));
        }

        let expansion = match node.expansion.get() {
            Some(expansion) => expansion,
            None => {
                // This is a leaf node.
                if node.visits.load(Ordering::SeqCst) < self.options.rollouts_before_expanding {
                    // Just rollout from here.
                    return node.update_stats(self.rollout(state));
                } else {
                    // Expand this node, and force a rollout when we recurse.
                    force_rollout = true;
                    node.expansion.try_set(new_expansion::<G>(state))
                }
            }
        };

        if let Some(winner) = expansion.winner {
            return node.update_stats(match winner {
                Winner::PlayerJustMoved => 1,
                Winner::PlayerToMove => -1,
                Winner::Draw => 0,
            });
        }

        // Recurse.
        let next = node.best_child(1.).unwrap();
        let m = next.m.as_ref().unwrap();
        m.apply(state);
        let result = -self.simulate(next, state, force_rollout)?;
        m.undo(state);

        // Backpropagate.
        node.update_stats(result)
    }
}

impl<G: Game> Strategy<G> for MonteCarloTreeSearch<G>
where
    G: Send + 'static,
    G::S: Clone + Send + 'static,
    G::M: Copy + Send + Sync + 'static,
{
    fn choose_move(&mut self, s: &G::S) -> Option<G::M> {
        let root = Arc::new(Node::<G::M>::new(None));
        root.expansion.try_set(new_expansion::<G>(s));

        let num_threads = self.options.num_threads.unwrap_or_else(num_cpus::get) as u32;
        let (rollouts_per_thread, extra) = if self.max_rollouts == 0 {
            (u32::MAX, 0)
        } else {
            let rollouts_per_thread = self.max_rollouts / num_threads;
            (rollouts_per_thread, self.max_rollouts - rollouts_per_thread * num_threads)
        };
        self.timeout = if self.max_time == Duration::default() {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.max_time)
        };

        let threads = (1..num_threads)
            .map(|_| {
                let node = root.clone();
                let mut state = s.clone();
                let mcts = self.clone();
                spawn(move || {
                    for _ in 0..rollouts_per_thread {
                        if mcts.simulate(&node, &mut state, false).is_none() {
                            break;
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        let mut state = s.clone();
        for _ in 0..rollouts_per_thread + extra {
            if self.simulate(&root, &mut state, false).is_none() {
                break;
            }
        }

        // Wait for threads.
        for thread in threads {
            thread.join().unwrap();
        }

        let exploration = 0.0; // Just get best node.
        root.best_child(exploration).map(|node| node.m.unwrap())
    }

    fn set_timeout(&mut self, timeout: Duration) {
        self.max_rollouts = 0;
        self.max_time = timeout;
    }

    fn set_max_depth(&mut self, depth: u8) {
        // Set some arbitrary function of rollouts.
        self.max_time = Duration::default();
        self.max_rollouts = depth as u32 * 100;
    }
}
