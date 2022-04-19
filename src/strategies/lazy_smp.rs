//! An implementation of parallelized Negamax via the Lazy Symmetric
//! MultiProcessing algorithm.
//!
//! This parallel algorithm minimizes cross-thread synchronization and
//! minimizes game state cloning, at the expense of doing more duplicative
//! work across different threads.

extern crate num_cpus;
extern crate rand;

use super::super::interface::*;
use super::iterative::Negamaxer;
use super::table::*;
use super::util::*;

use rand::seq::SliceRandom;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::spawn;
use std::time::{Duration, Instant};

/// Options to use for the iterative search engine.
#[derive(Clone, Copy)]
pub struct LazySmpOptions {
    table_byte_size: usize,
    step_increment: u8,
    max_quiescence_depth: u8,
    aspiration_window: Option<Evaluation>,
    // Default is one per core.
    num_threads: Option<usize>,
    // TODO: optional bonus thread local TT?
    // TODO: min_TT_depth?
    // TODO: alternating depths in alternating threads
}

impl LazySmpOptions {
    pub fn new() -> Self {
        LazySmpOptions {
            table_byte_size: 32_000_000,
            step_increment: 1,
            max_quiescence_depth: 0,
            aspiration_window: None,
            num_threads: None,
        }
    }
}

impl Default for LazySmpOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl LazySmpOptions {
    /// Approximately how large the transposition table should be in memory.
    pub fn with_table_byte_size(mut self, size: usize) -> Self {
        self.table_byte_size = size;
        self
    }

    /// Increment the depth by two between iterations.
    pub fn with_double_step_increment(mut self) -> Self {
        self.step_increment = 2;
        self
    }

    /// Enable [quiescence
    /// search](https://en.wikipedia.org/wiki/Quiescence_search) at the leaves
    /// of the search tree.  The Game must implement `generate_noisy_moves`
    /// for the search to know when the state has become "quiet".
    pub fn with_quiescence_search_depth(mut self, depth: u8) -> Self {
        self.max_quiescence_depth = depth;
        self
    }

    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Whether to search first in a narrow window around the previous root
    /// value on each iteration.
    pub fn with_aspiration_window(mut self, window: Evaluation) -> Self {
        self.aspiration_window = Some(window);
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

struct Helper<E: Evaluator>
where
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy + Eq,
{
    negamaxer: Negamaxer<E, Arc<ConcurrentTable<<E::G as Game>::M>>>,
    command: Arc<Mutex<Command<<E::G as Game>::S>>>,
    waiter: Arc<Condvar>,
}

impl<E: Evaluator> Helper<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn process(&mut self) {
        loop {
            let mut search = {
                let command = self.command.lock().unwrap();
                let command =
                    self.waiter.wait_while(command, |c| matches!(*c, Command::Wait)).unwrap();
                match *command {
                    Command::Exit => return,
                    Command::Wait => continue,
                    Command::Search(ref search) => search.clone(),
                }
            };

            self.negamaxer.set_timeout(search.timeout.clone());
            let mut alpha = WORST_EVAL;
            let mut beta = BEST_EVAL;
            self.negamaxer.table.check(
                search.state.zobrist_hash(),
                search.depth,
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
                if let Some(value) =
                    self.negamaxer.negamax(&mut search.state, search.depth, alpha, beta)
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
    table: Arc<ConcurrentTable<<E::G as Game>::M>>,
    negamaxer: Negamaxer<E, Arc<ConcurrentTable<<E::G as Game>::M>>>,
    command: Arc<Mutex<Command<<E::G as Game>::S>>>,
    signal: Arc<Condvar>,

    opts: LazySmpOptions,

    // Runtime stats for the last move generated.
    prev_value: Evaluation,
    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored at each depth.
    nodes_explored: Vec<u64>,
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
    pub fn new(eval: E, opts: LazySmpOptions) -> LazySmp<E>
    where
        E: 'static,
    {
        let table = Arc::new(ConcurrentTable::new(opts.table_byte_size));
        let command = Arc::new(Mutex::new(Command::Wait));
        let signal = Arc::new(Condvar::new());
        // start n-1 helper threads
        for _ in 1..opts.num_threads.unwrap_or_else(num_cpus::get) {
            let table2 = table.clone();
            let eval2 = eval.clone();
            let command2 = command.clone();
            let waiter = signal.clone();
            spawn(move || {
                let mut helper = Helper {
                    negamaxer: Negamaxer::new(table2, eval2, opts.max_quiescence_depth, true),
                    command: command2,
                    waiter,
                };
                helper.process();
            });
        }
        let negamaxer = Negamaxer::new(table.clone(), eval, opts.max_quiescence_depth, true);
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

    // TODO: gather stats from helper threads.
    // Return a human-readable summary of the last move generation.
    //pub fn stats(&self) -> String {
    //}

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

        let mut depth = self.max_depth as u8 % self.opts.step_increment;
        while depth <= self.max_depth as u8 {
            // First, a serial aspiration search to at least establish some bounds.
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

            let entry = self.table.lookup(root_hash).unwrap();
            best_move = Some(entry.best_move());

            self.actual_depth = max(self.actual_depth, depth);
            self.prev_value = entry.value;
            depth += self.opts.step_increment;
            self.table.populate_pv(&mut self.pv, &mut s_clone, depth + 1);
        }
        self.wall_time = start_time.elapsed();
        best_move
    }
}
