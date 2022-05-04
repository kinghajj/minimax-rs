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
}

#[derive(Clone)]
struct Search<S: Clone> {
    state: S,
    depth: u8,
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
    command: Arc<Mutex<Command<<E::G as Game>::S>>>,
    waiter: Arc<Condvar>,
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
        loop {
            let mut search = {
                let command = self.command.lock().unwrap();
                // Stay waiting during Wait command or if we already completed Search command.
                let command = self
                    .waiter
                    .wait_while(command, |c| match *c {
                        Command::Exit => false,
                        Command::Wait => true,
                        Command::Search(ref search) => {
                            search.state.zobrist_hash() == prev_hash && search.depth == prev_depth
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

            let depth = search.depth + self.extra_depth;
            self.negamaxer.set_timeout(search.timeout.clone());
            let mut alpha = WORST_EVAL;
            let mut beta = BEST_EVAL;
            self.negamaxer.table.check(
                search.state.zobrist_hash(),
                depth,
                &mut None,
                &mut alpha,
                &mut beta,
            );

            // Randomize the first level of moves.
            let mut moves = Vec::new();
            E::G::generate_moves(&search.state, &mut moves);
            moves.shuffle(&mut rand::thread_rng());
            // Negamax search the rest.
            for m in moves {
                m.apply(&mut search.state);
                if let Some(value) = self.negamaxer.negamax(&mut search.state, depth, alpha, beta) {
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

pub struct LazySmp<E: Evaluator>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    max_depth: usize,
    max_time: Duration,
    table: Arc<LockfreeTable<<E::G as Game>::M>>,
    negamaxer: Negamaxer<E, Arc<LockfreeTable<<E::G as Game>::M>>>,
    command: Arc<Mutex<Command<<E::G as Game>::S>>>,
    signal: Arc<Condvar>,

    opts: IterativeOptions,

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

impl<E: Evaluator> Drop for LazySmp<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn drop(&mut self) {
        *self.command.lock().unwrap() = Command::Exit;
        self.signal.notify_all();
    }
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
        let command = Arc::new(Mutex::new(Command::Wait));
        let signal = Arc::new(Condvar::new());
        let stats = Arc::new(SharedStats::new());
        // start n-1 helper threads
        for iter in 1..smp_opts.num_threads.unwrap_or_else(num_cpus::get) {
            let table2 = table.clone();
            let eval2 = eval.clone();
            let opts2 = opts.clone();
            let command2 = command.clone();
            let waiter = signal.clone();
            let stats2 = stats.clone();
            let extra_depth = if smp_opts.differing_depths { iter as u8 & 1 } else { 0 };
            spawn(move || {
                let mut helper = Helper {
                    negamaxer: Negamaxer::new(table2, eval2, opts2),
                    command: command2,
                    waiter,
                    stats: stats2,
                    extra_depth,
                };
                helper.process();
            });
        }
        let negamaxer = Negamaxer::new(table.clone(), eval, opts.clone());
        LazySmp {
            max_depth: 100,
            max_time: Duration::from_secs(5),
            table,
            negamaxer,
            command,
            signal,
            prev_value: 0,
            opts,
            actual_depth: 0,
            nodes_explored: Vec::new(),
            shared_stats: stats,
            pv: Vec::new(),
            wall_time: Duration::default(),
        }
    }

    /// Set the maximum depth to search. Disables the timeout.
    /// This can be changed between moves while reusing the transposition table.
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
        self.max_time = Duration::new(0, 0);
    }

    /// Set the maximum time to compute the best move. When the timeout is
    /// hit, it returns the best move found of the previous full
    /// iteration. Unlimited max depth.
    pub fn set_timeout(&mut self, max_time: Duration) {
        self.max_time = max_time;
        self.max_depth = 100;
    }

    /// Return a human-readable summary of the last move generation.
    pub fn stats(&self) -> String {
        let total_nodes_explored: u64 = self.nodes_explored.iter().sum();
        let mean_branching_factor = self.shared_stats.generated_moves.load(Ordering::SeqCst) as f64
            / self.shared_stats.generate_move_calls.load(Ordering::SeqCst) as f64;
        let effective_branching_factor = (*self.nodes_explored.last().unwrap_or(&0) as f64)
            .powf((self.actual_depth as f64 + 1.0).recip());
        let throughput = (total_nodes_explored + self.negamaxer.nodes_explored) as f64
            / self.wall_time.as_secs_f64();
        format!("Explored {} nodes to depth {}. MBF={:.1} EBF={:.1}\nPartial exploration of next depth hit {} nodes.\n{} nodes/sec",
		total_nodes_explored, self.actual_depth, mean_branching_factor, effective_branching_factor,
		self.negamaxer.nodes_explored, throughput as usize)
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }

    /// Return what the engine considered to be the best sequence of moves
    /// from both sides.
    pub fn principal_variation(&self) -> &[<E::G as Game>::M] {
        &self.pv[..]
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
        let mut interval_start = start_time;

        let mut depth = self.max_depth as u8 % self.opts.step_increment;
        while depth <= self.max_depth as u8 {
            // First, a serial aspiration search to at least establish some bounds.
            if self.opts.verbose {
                interval_start = Instant::now();
                println!("LazySmp search depth {} around {}", depth + 1, self.prev_value);
            }
            if self
                .negamaxer
                .aspiration_search(
                    &mut s_clone,
                    depth + 1,
                    self.prev_value,
                    self.opts.aspiration_window.unwrap_or(2),
                )
                .is_none()
            {
                // Timeout.
                break;
            }
            if self.opts.verbose {
                let mut alpha = WORST_EVAL;
                let mut beta = BEST_EVAL;
                self.negamaxer.table.check(root_hash, depth + 1, &mut None, &mut alpha, &mut beta);
                let end = Instant::now();
                let interval = end - interval_start;
                println!(
                    "LazySmp aspiration search took {}ms; within bounds {}:{}",
                    interval.as_millis(),
                    alpha,
                    beta
                );
                interval_start = end;
            }

            let iteration_done = Arc::new(AtomicBool::new(false));
            {
                let mut command = self.command.lock().unwrap();
                *command = Command::Search(Search {
                    state: s.clone(),
                    depth,
                    timeout: iteration_done.clone(),
                });
                self.signal.notify_all();
            }

            let value = self.negamaxer.negamax(&mut s_clone, depth + 1, WORST_EVAL, BEST_EVAL);
            {
                *self.command.lock().unwrap() = Command::Wait;
            }
            iteration_done.store(true, Ordering::Relaxed);
            if value.is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }

            if self.opts.verbose {
                let interval = Instant::now() - interval_start;
                println!(
                    "LazySmp       full search took {}ms; returned {:?}",
                    interval.as_millis(),
                    value.unwrap()
                );
            }

            let entry = self.table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            self.actual_depth = max(self.actual_depth, depth);
            self.prev_value = entry.value;
            depth += self.opts.step_increment;
            self.table.populate_pv(&mut self.pv, &mut s_clone, depth + 1);
            self.shared_stats.update(&mut self.negamaxer);
            self.nodes_explored.push(self.shared_stats.reset_nodes_explored());
        }
        self.wall_time = start_time.elapsed();
        best_move
    }
}
