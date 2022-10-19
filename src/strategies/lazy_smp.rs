//! An implementation of parallelized Negamax via the Lazy Symmetric
//! MultiProcessing algorithm.
//!
//! This parallel algorithm minimizes cross-thread synchronization and
//! minimizes game state cloning, at the expense of doing more duplicative
//! work across different threads.

extern crate num_cpus;
extern crate rand;

use super::super::interface::*;
use super::iterative::{IterativeOptions, Negamaxer};
use super::sync_util::timeout_signal;
use super::table::*;
use super::util::*;

use rand::seq::SliceRandom;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::spawn;
use std::time::{Duration, Instant};

/// Options to use for the iterative search engine.
#[derive(Clone, Copy)]
pub struct LazySmpOptions {
    // Default is one per core.
    num_threads: Option<usize>,
    differing_depths: bool,
    // TODO: optional bonus thread local TT?
    // TODO: min_TT_depth?
}

impl LazySmpOptions {
    pub fn new() -> Self {
        LazySmpOptions { num_threads: None, differing_depths: false }
    }
}

impl Default for LazySmpOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl LazySmpOptions {
    /// Set the total number of threads to use. Otherwise defaults to num_cpus.
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enables the helper threads to explore the tree at multiple depths simultaneously.
    pub fn with_differing_depths(mut self) -> Self {
        self.differing_depths = true;
        self
    }

    fn num_threads(self) -> usize {
        self.num_threads.unwrap_or_else(num_cpus::get)
    }
}

#[derive(Clone)]
struct Search<S: Clone> {
    state: S,
    depth: u8,
    alpha: Evaluation,
    beta: Evaluation,
    timeout: Arc<AtomicBool>,
}

// A directive to the helper threads.
enum Command<S: Clone> {
    Wait,
    Exit,
    Search(Search<S>),
}

struct SharedStats {
    nodes_explored: AtomicU64,
    generated_moves: AtomicU64,
    generate_move_calls: AtomicU64,
}

impl SharedStats {
    fn new() -> Self {
        Self {
            nodes_explored: AtomicU64::new(0),
            generated_moves: AtomicU64::new(0),
            generate_move_calls: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.nodes_explored.store(0, Ordering::SeqCst);
        self.generated_moves.store(0, Ordering::SeqCst);
        self.generate_move_calls.store(0, Ordering::SeqCst);
    }

    fn update<E: Evaluator, T>(&self, negamaxer: &mut Negamaxer<E, T>) {
        self.nodes_explored.fetch_add(negamaxer.nodes_explored, Ordering::SeqCst);
        negamaxer.nodes_explored = 0;
        self.generated_moves.fetch_add(negamaxer.total_generated_moves, Ordering::SeqCst);
        negamaxer.total_generated_moves = 0;
        self.generate_move_calls.fetch_add(negamaxer.total_generate_move_calls, Ordering::SeqCst);
        negamaxer.total_generate_move_calls = 0;
    }

    fn reset_nodes_explored(&self) -> u64 {
        self.nodes_explored.swap(0, Ordering::SeqCst)
    }
}

struct Helper<E: Evaluator>
where
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy + Eq,
{
    negamaxer: Negamaxer<E, Arc<LockfreeTable<<E::G as Game>::M>>>,
    signal: Arc<CommandSignal<<E::G as Game>::S>>,
    stats: Arc<SharedStats>,
    extra_depth: u8,
}

impl<E: Evaluator> Helper<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn process(&mut self) {
        let mut prev_hash: u64 = 0;
        let mut prev_depth: u8 = 200;
        let mut prev_alpha = 0;
        let mut prev_beta = 0;
        loop {
            let mut search = {
                let command = self.signal.command.lock().unwrap();
                // Stay waiting during Wait command or if we already completed Search command.
                let command = self
                    .signal
                    .signal
                    .wait_while(command, |c| match *c {
                        Command::Exit => false,
                        Command::Wait => true,
                        Command::Search(ref search) => {
                            search.state.zobrist_hash() == prev_hash
                                && search.depth == prev_depth
                                && prev_alpha == search.alpha
                                && prev_beta == search.beta
                        }
                    })
                    .unwrap();
                // Do command.
                match *command {
                    Command::Exit => return,
                    Command::Wait => continue,
                    Command::Search(ref search) => search.clone(),
                }
            };
            prev_hash = search.state.zobrist_hash();
            prev_depth = search.depth;
            prev_alpha = search.alpha;
            prev_beta = search.beta;

            let depth = search.depth + self.extra_depth;
            self.negamaxer.set_timeout(search.timeout.clone());
            let mut alpha = search.alpha;
            let mut beta = search.beta;
            self.negamaxer.table.check(
                search.state.zobrist_hash(),
                depth,
                &mut None,
                &mut alpha,
                &mut beta,
            );

            self.negamaxer.countermoves.advance_generation(E::G::null_move(&search.state));
            // Randomize the first level of moves.
            let mut moves = Vec::new();
            E::G::generate_moves(&search.state, &mut moves);
            moves.shuffle(&mut rand::thread_rng());
            // Negamax search the rest.
            for m in moves {
                m.apply(&mut search.state);
                if let Some(value) =
                    self.negamaxer.negamax(&mut search.state, Some(m), depth - 1, alpha, beta)
                {
                    alpha = max(alpha, -value);
                } else {
                    break;
                }
                if alpha >= beta {
                    break;
                }
                m.undo(&mut search.state);
            }

            // Computation finished or interrupted, go back to sleep.
            self.stats.update(&mut self.negamaxer);
        }
    }
}

struct CommandSignal<S: Clone> {
    command: Mutex<Command<S>>,
    signal: Condvar,
}

impl<S> CommandSignal<S>
where
    S: Clone,
{
    fn new() -> Self {
        Self { command: Mutex::new(Command::Wait), signal: Condvar::new() }
    }

    fn update(&self, new_command: Command<S>) {
        let mut command = self.command.lock().unwrap();
        if let Command::Search(ref search) = *command {
            search.timeout.store(true, Ordering::SeqCst);
        }
        *command = new_command;
        self.signal.notify_all();
    }

    fn wait(&self) {
        self.update(Command::Wait);
    }

    fn new_search(&self, state: &S, depth: u8, alpha: Evaluation, beta: Evaluation) {
        self.update(Command::Search(Search {
            state: state.clone(),
            depth,
            alpha,
            beta,
            timeout: Arc::new(AtomicBool::new(false)),
        }));
    }
}

impl<S> Drop for CommandSignal<S>
where
    S: Clone,
{
    fn drop(&mut self) {
        self.update(Command::Exit);
    }
}

pub struct LazySmp<E: Evaluator>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    max_depth: u8,
    max_time: Duration,
    table: Arc<LockfreeTable<<E::G as Game>::M>>,
    negamaxer: Negamaxer<E, Arc<LockfreeTable<<E::G as Game>::M>>>,
    signal: Arc<CommandSignal<<E::G as Game>::S>>,

    opts: IterativeOptions,
    num_threads: usize,

    // Runtime stats for the last move generated.
    prev_value: Evaluation,
    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored at each depth.
    nodes_explored: Vec<u64>,
    shared_stats: Arc<SharedStats>,
    pv: Vec<<E::G as Game>::M>,
    wall_time: Duration,
}

impl<E: Evaluator> LazySmp<E>
where
    <E::G as Game>::S: Clone + Zobrist + Send,
    <E::G as Game>::M: Copy + Eq + Send,
    E: Clone + Send,
{
    pub fn new(eval: E, opts: IterativeOptions, smp_opts: LazySmpOptions) -> LazySmp<E>
    where
        E: 'static,
    {
        let table = Arc::new(LockfreeTable::new(opts.table_byte_size));
        let stats = Arc::new(SharedStats::new());
        let signal = Arc::new(CommandSignal::new());
        let num_threads = smp_opts.num_threads();
        // start n-1 helper threads
        for iter in 1..num_threads {
            let table2 = table.clone();
            let eval2 = eval.clone();
            let opts2 = opts;
            let signal2 = signal.clone();
            let stats2 = stats.clone();
            let extra_depth = if smp_opts.differing_depths { iter as u8 & 1 } else { 0 };
            spawn(move || {
                let mut helper = Helper {
                    negamaxer: Negamaxer::new(table2, eval2, opts2),
                    signal: signal2,
                    stats: stats2,
                    extra_depth,
                };
                helper.process();
            });
        }
        let negamaxer = Negamaxer::new(table.clone(), eval, opts);
        LazySmp {
            max_depth: 99,
            max_time: Duration::from_secs(5),
            table,
            negamaxer,
            signal,
            prev_value: 0,
            opts,
            num_threads,
            actual_depth: 0,
            nodes_explored: Vec::new(),
            shared_stats: stats,
            pv: Vec::new(),
            wall_time: Duration::default(),
        }
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }
}

impl<E: Evaluator> LazySmp<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    /// Return a human-readable summary of the last move generation.
    pub fn stats(&self, s: &mut <E::G as Game>::S) -> String {
        let total_nodes_explored: u64 = self.nodes_explored.iter().sum();
        let mean_branching_factor = self.shared_stats.generated_moves.load(Ordering::SeqCst) as f64
            / self.shared_stats.generate_move_calls.load(Ordering::SeqCst) as f64;
        let effective_branching_factor = (*self.nodes_explored.last().unwrap_or(&0) as f64)
            .powf((self.actual_depth as f64 + 1.0).recip());
        let throughput = (total_nodes_explored + self.negamaxer.nodes_explored) as f64
            / self.wall_time.as_secs_f64();
        format!("Principal variation: {}\nExplored {} nodes to depth {}. MBF={:.1} EBF={:.1}\nPartial exploration of next depth hit {} nodes.\n{} nodes/sec",
                pv_string::<E::G>(&self.pv[..], s),
		total_nodes_explored, self.actual_depth, mean_branching_factor, effective_branching_factor,
		self.negamaxer.nodes_explored, throughput as usize)
    }
}

impl<E: Evaluator> Strategy<E::G> for LazySmp<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        self.table.concurrent_advance_generation();
        // Reset stats.
        self.shared_stats.reset();
        self.nodes_explored.clear();
        self.actual_depth = 0;
        let start_time = Instant::now();
        // Start timer if configured.
        self.negamaxer.set_timeout(if self.max_time == Duration::new(0, 0) {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.max_time)
        });

        let root_hash = s.zobrist_hash();
        let mut s_clone = s.clone();
        let mut best_move = None;
        let mut interval_start;
        let mut maxxed = false;
        // Store the moves so they can be reordered every iteration.
        let mut moves = Vec::new();
        E::G::generate_moves(&s_clone, &mut moves);
        // Start in a random order.
        moves.shuffle(&mut rand::thread_rng());
        let mut moves = moves.into_iter().map(|m| ValueMove::new(0, m)).collect::<Vec<_>>();

        let mut depth = self.max_depth as u8 % self.opts.step_increment;
        if depth == 0 {
            depth = self.opts.step_increment;
        }
        while depth <= self.max_depth as u8 {
            interval_start = Instant::now();
            if let Some(window) = self.opts.aspiration_window {
                // First, parallel aspiration search to at least establish some bounds.
                let mut alpha = self.prev_value.saturating_sub(window);
                if alpha < WORST_EVAL {
                    alpha = WORST_EVAL;
                }
                let beta = self.prev_value.saturating_add(window);
                self.signal.new_search(s, depth, alpha, beta);

                if self
                    .negamaxer
                    .aspiration_search(&mut s_clone, depth, self.prev_value, window)
                    .is_none()
                {
                    // Timeout.
                    break;
                }
                if self.opts.verbose && !maxxed {
                    if let Some(entry) = self.table.lookup(root_hash) {
                        let end = Instant::now();
                        let interval = end - interval_start;
                        eprintln!(
                            "LazySmp (threads={}) aspiration depth{:>2} took{:>5}ms; bounds{:>5} bestmove={}",
                            self.num_threads,
                            depth,
                            interval.as_millis(),
                            entry.bounds(),
                            move_id::<E::G>(&mut s_clone, entry.best_move)
                        );
                        interval_start = end;
                    }
                }
            }

            self.signal.new_search(s, depth, WORST_EVAL, BEST_EVAL);

            let value = self.negamaxer.search_and_reorder(&mut s_clone, &mut moves, depth);
            if value.is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }

            let entry = self.table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            if self.opts.verbose && !maxxed {
                let interval = Instant::now() - interval_start;
                eprintln!(
                    "LazySmp (threads={}) fullsearch depth{:>2} took{:>5}ms; value{:>6} bestmove={}",
                    self.num_threads,
                    depth,
                    interval.as_millis(),
                    entry.value_string(),
                    move_id::<E::G>(&mut s_clone, entry.best_move)
                );
                if unclamp_value(value.unwrap()).abs() == BEST_EVAL {
                    maxxed = true;
                }
            }

            self.actual_depth = max(self.actual_depth, depth);
            self.prev_value = entry.value;
            depth += self.opts.step_increment;
            self.table.populate_pv(&mut self.pv, &mut s_clone);
            self.shared_stats.update(&mut self.negamaxer);
            self.nodes_explored.push(self.shared_stats.reset_nodes_explored());
        }
        self.signal.wait();
        self.wall_time = start_time.elapsed();
        if self.opts.verbose {
            let mut s_clone = s.clone();
            eprintln!("{}", self.stats(&mut s_clone));
        }
        best_move
    }

    fn set_timeout(&mut self, max_time: Duration) {
        self.max_time = max_time;
        self.max_depth = 99;
    }

    fn set_max_depth(&mut self, depth: u8) {
        self.max_depth = depth;
        self.max_time = Duration::new(0, 0);
    }

    fn principal_variation(&self) -> Vec<<E::G as Game>::M> {
        self.pv.clone()
    }
}
