use super::super::interface::*;
use super::super::util::AppliedMove;
use super::sync_util::*;
use super::util::{move_id, pv_string, random_best};

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::marker::PhantomData;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const WIN: i32 = i32::MAX;
// Make sure they negate to each other, unlike i32::MIN.
const LOSS: i32 = -WIN;

struct Node<M> {
    // The Move to get from the parent to here.
    // Only None at the root.
    m: Option<M>,
    visits: AtomicU32,
    // +1 for wins, -1 for losses, +0 for draws.
    // From perspective of the player that made this move.
    score: AtomicI32,
    // Lazily populated if this node guarantees a particular end state.
    // WIN for a guaranteed win, LOSS for a guaranteed loss.
    // Not bothering with draws.
    winner: AtomicI32,
    // Lazily populated.
    expansion: AtomicBox<NodeExpansion<M>>,
}

struct NodeExpansion<M> {
    children: Vec<Node<M>>,
}

fn new_expansion<G: Game>(state: &G::S) -> Box<NodeExpansion<G::M>> {
    let mut moves = Vec::new();
    G::generate_moves(state, &mut moves);
    let children = moves.into_iter().map(|m| Node::new(Some(m))).collect::<Vec<_>>();
    Box::new(NodeExpansion { children })
}

impl<M> Node<M> {
    fn new(m: Option<M>) -> Self {
        Node {
            m,
            expansion: AtomicBox::default(),
            visits: AtomicU32::new(0),
            score: AtomicI32::new(0),
            winner: AtomicI32::new(0),
        }
    }

    // Choose best child based on UCT.
    fn best_child(&self, exploration_score: f32) -> Option<&Node<M>> {
        let mut log_visits = (self.visits.load(SeqCst) as f32).log2();
        // Keep this numerator non-negative.
        if log_visits < 0.0 {
            log_visits = 0.0;
        }

        let expansion = self.expansion.get()?;
        random_best(expansion.children.as_slice(), |node| {
            node.uct_score(exploration_score, log_visits)
        })
    }

    fn uct_score(&self, exploration_score: f32, log_parent_visits: f32) -> f32 {
        let winner = self.winner.load(Relaxed);
        if winner < 0 {
            // Large enough to be returned from best_move, smaller than any other value.
            // This effectively ignores any moves that we've proved guarantee losses.
            // The MCTS-Solver paper says not to do this, but I don't buy their argument.
            // Those moves effectivey won't exist in our search, and we'll
            // have to see if the remaining moves make the parent moves worthwhile.
            return -1.0;
        }
        if winner > 0 {
            return f32::INFINITY;
        }
        let visits = self.visits.load(Relaxed) as f32;
        let score = self.score.load(Relaxed) as f32;
        if visits == 0.0 {
            // Avoid NaNs.
            return if exploration_score > 0.0 { f32::INFINITY } else { 0.0 };
        }
        let win_ratio = (score + visits) / (2.0 * visits);
        win_ratio + exploration_score * (2.0 * log_parent_visits / visits).sqrt()
    }

    fn pre_update_stats(&self) {
        // Use a technicque called virtual loss to assume we've lost any
        // ongoing simulation to bias concurrent threads against exploring it.
        self.visits.fetch_add(1, SeqCst);
        self.score.fetch_add(-1, SeqCst);
    }

    fn update_stats(&self, result: i32) -> Option<i32> {
        if result == WIN || result == LOSS {
            self.winner.store(result, SeqCst);
        } else {
            // Adjust for virtual loss.
            self.score.fetch_add(result + 1, SeqCst);
        }
        // Always return Some, as we aren't timed out.
        Some(result)
    }
}

/// Options for MonteCarloTreeSearch.
#[derive(Clone)]
pub struct MCTSOptions {
    pub verbose: bool,
    max_rollout_depth: u32,
    rollouts_before_expanding: u32,
    // None means use num_cpus.
    num_threads: Option<usize>,
}

impl Default for MCTSOptions {
    fn default() -> Self {
        Self {
            verbose: false,
            max_rollout_depth: 100,
            rollouts_before_expanding: 0,
            num_threads: None,
        }
    }
}

impl MCTSOptions {
    /// Enable verbose print statements after each search.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

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

/// Advanced random rollout policy for Monte Carlo Tree Search.
pub trait RolloutPolicy {
    /// The type of game that can be evaluated.
    type G: Game;

    /// Custom function to choose random move during rollouts.
    /// Implementations can bias towards certain moves, ensure winning moves, etc.
    /// The provided move vec is for scratch space.
    fn random_move(
        &self, state: &mut <Self::G as Game>::S, move_scratch: &mut Vec<<Self::G as Game>::M>,
        rng: &mut ThreadRng,
    ) -> <Self::G as Game>::M;

    /// Implementation of a rollout over many random moves. Not needed to be overridden.
    fn rollout(&self, options: &MCTSOptions, state: &<Self::G as Game>::S) -> i32
    where
        <Self::G as Game>::S: Clone,
    {
        let mut rng = rand::thread_rng();
        let mut depth = options.max_rollout_depth;
        let mut state = state.clone();
        let mut moves = Vec::new();
        let mut sign = 1;
        loop {
            if let Some(winner) = Self::G::get_winner(&state) {
                let first = depth == options.max_rollout_depth;
                return match winner {
                    Winner::PlayerJustMoved => {
                        if first {
                            WIN
                        } else {
                            1
                        }
                    }
                    Winner::PlayerToMove => {
                        if first {
                            LOSS
                        } else {
                            -1
                        }
                    }
                    Winner::Draw => 0,
                } * sign;
            }

            if depth == 0 {
                return 0;
            }

            moves.clear();
            let m = self.random_move(&mut state, &mut moves, &mut rng);
            if let Some(new_state) = Self::G::apply(&mut state, m) {
                state = new_state;
            }
            sign = -sign;
            depth -= 1;
        }
    }
}

struct DumbRolloutPolicy<G: Game> {
    game_type: PhantomData<G>,
}

impl<G: Game> RolloutPolicy for DumbRolloutPolicy<G> {
    type G = G;
    fn random_move(
        &self, state: &mut <Self::G as Game>::S, moves: &mut Vec<<Self::G as Game>::M>,
        rng: &mut ThreadRng,
    ) -> <Self::G as Game>::M {
        G::generate_moves(state, moves);
        *moves.choose(rng).unwrap()
    }
}

/// A strategy that uses random playouts to explore the game tree to decide on the best move.
/// This can be used without an Evaluator, just using the rules of the game.
pub struct MonteCarloTreeSearch<G: Game> {
    options: MCTSOptions,
    max_rollouts: u32,
    max_time: Duration,
    timeout: Arc<AtomicBool>,
    rollout_policy: Option<Box<dyn RolloutPolicy<G = G> + Sync>>,
    pv: Vec<G::M>,
    game_type: PhantomData<G>,
}

impl<G: Game> MonteCarloTreeSearch<G> {
    pub fn new(options: MCTSOptions) -> Self {
        Self {
            options,
            max_rollouts: 0,
            max_time: Duration::from_secs(5),
            timeout: Arc::new(AtomicBool::new(false)),
            rollout_policy: None,
            pv: Vec::new(),
            game_type: PhantomData,
        }
    }

    /// Create a searcher with a custom rollout policy. You could bias the
    /// random move generation to prefer certain kinds of moves, always choose
    /// winning moves, etc.
    pub fn new_with_policy(
        options: MCTSOptions, policy: Box<dyn RolloutPolicy<G = G> + Sync>,
    ) -> Self {
        Self {
            options,
            max_rollouts: 0,
            max_time: Duration::from_secs(5),
            timeout: Arc::new(AtomicBool::new(false)),
            rollout_policy: Some(policy),
            pv: Vec::new(),
            game_type: PhantomData,
        }
    }

    /// Instead of a timeout, run this many rollouts to choose a move.
    pub fn set_max_rollouts(&mut self, rollouts: u32) {
        self.max_time = Duration::default();
        self.max_rollouts = rollouts;
    }

    fn rollout(&self, state: &G::S) -> i32
    where
        G: Sync,
        G::S: Clone,
    {
        self.rollout_policy.as_ref().map(|p| p.rollout(&self.options, state)).unwrap_or_else(|| {
            DumbRolloutPolicy::<G> { game_type: PhantomData }.rollout(&self.options, state)
        })
    }

    // Explore the tree, make a new node, rollout, backpropagate.
    fn simulate(&self, node: &Node<G::M>, state: &mut G::S, mut force_rollout: bool) -> Option<i32>
    where
        G: Sync,
        G::S: Clone,
    {
        if self.timeout.load(Relaxed) {
            return None;
        }
        let winner = node.winner.load(Relaxed);
        if winner != 0 {
            return Some(winner);
        }
        node.pre_update_stats();

        if force_rollout {
            return node.update_stats(self.rollout(state));
        }

        let expansion = match node.expansion.get() {
            Some(expansion) => expansion,
            None => {
                // This is a leaf node.
                if node.visits.load(SeqCst) <= self.options.rollouts_before_expanding {
                    // Just rollout from here.
                    return node.update_stats(self.rollout(state));
                } else {
                    // Check for terminal node.
                    match G::get_winner(state) {
                        Some(Winner::PlayerJustMoved) => return node.update_stats(WIN),
                        Some(Winner::PlayerToMove) => return node.update_stats(LOSS),
                        Some(Winner::Draw) => return node.update_stats(0),
                        _ => {}
                    }
                    // Expand this node, and force a rollout when we recurse.
                    force_rollout = true;
                    node.expansion.try_set(new_expansion::<G>(state))
                }
            }
        };

        // Recurse.
        let next = match node.best_child(1.) {
            Some(child) => child,
            // TODO: Weird race condition?
            None => return Some(0),
        };
        let m = next.m.as_ref().unwrap();
        let mut new = AppliedMove::<G>::new(state, *m);
        let child_result = self.simulate(next, &mut new, force_rollout)?;

        // Propagate up forced wins and losses.
        let result = if child_result == WIN {
            // Having a guaranteed win child makes you a loser parent.
            LOSS
        } else if child_result == LOSS {
            // Having all guaranteed loser children makes you a winner parent.
            if expansion.children.iter().all(|node| node.winner.load(Relaxed) == LOSS) {
                WIN
            } else {
                -1
            }
        } else {
            -child_result
        };

        // Backpropagate.
        node.update_stats(result)
    }
}

impl<G: Game> Strategy<G> for MonteCarloTreeSearch<G>
where
    G: Sync,
    G::S: Clone + Send,
    G::M: Copy + Sync,
{
    fn choose_move(&mut self, s: &G::S) -> Option<G::M> {
        let start_time = Instant::now();
        let root = Box::new(Node::<G::M>::new(None));
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

        thread::scope(|scope| {
            for i in 0..num_threads {
                let node = &*root;
                let mtcs = &*self;
                let mut state = s.clone();
                scope.spawn(move || {
                    let rollouts = rollouts_per_thread + (i < extra) as u32;
                    for _ in 0..rollouts {
                        if mtcs.simulate(node, &mut state, false).is_none() {
                            break;
                        }
                    }
                });
            }
        });

        // Compute PV.
        self.pv.clear();
        let mut node = &*root;
        while let Some(best) = node.best_child(0.0) {
            self.pv.push(best.m.unwrap());
            node = best;
        }

        if self.options.verbose {
            let total_visits = root.visits.load(Relaxed);
            let duration = Instant::now().duration_since(start_time);
            let rate = total_visits as f64 / num_threads as f64 / duration.as_secs_f64();
            eprintln!(
                "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
                num_threads, total_visits, rate
            );
            // Sort moves by visit count, largest first.
            let mut children = root
                .expansion
                .get()?
                .children
                .iter()
                .map(|node| (node.visits.load(Relaxed), node.score.load(Relaxed), node.m))
                .collect::<Vec<_>>();
            children.sort_by_key(|t| !t.0);

            // Dump stats about the top 10 nodes.
            for (visits, score, m) in children.into_iter().take(10) {
                // Normalized so all wins is 100%, all draws is 50%, and all losses is 0%.
                let win_rate = (score as f64 + visits as f64) / (visits as f64 * 2.0);
                eprintln!(
                    "{:>6} visits, {:.02}% wins: {}",
                    visits,
                    win_rate * 100.0,
                    move_id::<G>(s, m)
                );
            }

            // Dump PV.
            eprintln!("Principal variation: {}", pv_string::<G>(&self.pv[..], s));
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
        self.max_rollouts = 5u32
            .saturating_pow(depth as u32)
            .saturating_mul(self.options.rollouts_before_expanding + 1);
    }

    fn principal_variation(&self) -> Vec<G::M> {
        self.pv.clone()
    }
}
