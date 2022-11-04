//! An implementation of iterative deeping, with each iteration executed in parallel.
//!
//! This implementation evaluates the best guess at each move first, then
//! parallelizes all other moves using rayon.
//!
//! This is based on the Young Brothers Wait Concept and CilkChess.

extern crate rayon;

use super::super::interface::*;
use super::super::util::*;
use super::iterative::{IterativeOptions, Stats};
use super::sync_util::{timeout_signal, CachePadded, ThreadLocal};
use super::table::*;
use super::util::*;

use rayon::prelude::*;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, AtomicI16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Options to use for the parallel search engine.
#[derive(Clone, Copy)]
pub struct ParallelOptions {
    pub num_threads: Option<usize>,
    serial_cutoff_depth: u8,
    pub background_pondering: bool,
}

impl ParallelOptions {
    pub fn new() -> Self {
        ParallelOptions { num_threads: None, serial_cutoff_depth: 1, background_pondering: false }
    }
}

impl Default for ParallelOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelOptions {
    /// Set the total number of threads to use. Otherwise defaults to num_cpus.
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// At what depth should we stop trying to parallelize and just run serially.
    pub fn with_serial_cutoff_depth(mut self, depth: u8) -> Self {
        self.serial_cutoff_depth = depth;
        self
    }

    /// Continuing processing during opponent's move.
    pub fn with_background_pondering(mut self) -> Self {
        self.background_pondering = true;
        self
    }

    pub fn num_threads(self) -> usize {
        self.num_threads.unwrap_or_else(num_cpus::get)
    }
}

struct ParallelNegamaxer<E: Evaluator> {
    table: Arc<LockfreeTable<<E::G as Game>::M>>,
    eval: E,
    opts: IterativeOptions,
    par_opts: ParallelOptions,
    timeout: Arc<AtomicBool>,
    stats: ThreadLocal<CachePadded<Stats>>,
    move_pool: ThreadLocal<MovePool<<E::G as Game>::M>>,
    countermoves: ThreadLocal<CounterMoves<<E::G as Game>::M>>,
    pv: Mutex<Vec<<E::G as Game>::M>>,
}

impl<E: Evaluator> ParallelNegamaxer<E>
where
    <E::G as Game>::S: Clone + Zobrist + Send + Sync,
    <E::G as Game>::M: Copy + Eq + Send + Sync,
    E: Clone + Sync + Send + 'static,
{
    fn new(
        opts: IterativeOptions, par_opts: ParallelOptions, eval: E,
        table: Arc<LockfreeTable<<E::G as Game>::M>>, timeout: Arc<AtomicBool>,
        thread_pool: &rayon::ThreadPool,
    ) -> Self {
        Self {
            table,
            eval,
            opts,
            par_opts,
            timeout,
            stats: ThreadLocal::new(CachePadded::default, thread_pool),
            move_pool: ThreadLocal::new(MovePool::default, thread_pool),
            countermoves: ThreadLocal::new(
                || CounterMoves::new(opts.countermove_table, opts.countermove_history_table),
                thread_pool,
            ),
            pv: Mutex::new(Vec::new()),
        }
    }

    fn principal_variation(&self) -> Vec<<E::G as Game>::M> {
        self.pv.lock().unwrap().clone()
    }

    fn null_move_check(
        &self, s: &mut <E::G as Game>::S, depth: u8, beta: Evaluation,
    ) -> Option<Evaluation> {
        if let (Some(depth_reduction), Some(null_move)) =
            (self.opts.null_move_depth, E::G::null_move(s))
        {
            // Default to a minimum of depth=1 after null moving.
            if depth > depth_reduction &&
	    // If the position already seems pretty awesome.
	      self.eval.evaluate(s) >= beta
            {
                // If we just pass and let the opponent play this position (at reduced depth),
                null_move.apply(s);
                let value = -self.negamax(s, None, depth - depth_reduction, -beta, -beta + 1)?;
                null_move.undo(s);
                // is the result still so good that we shouldn't bother with a full search?
                return Some(value);
            }
        }
        // If we didn't check, return a low value that won't trigger beta cutoff.
        Some(WORST_EVAL)
    }

    // Negamax only among noisy moves.
    fn noisy_negamax(
        &self, s: &mut <E::G as Game>::S, depth: u8, mut alpha: Evaluation, beta: Evaluation,
    ) -> Option<Evaluation> {
        if self.timeout.load(Ordering::Relaxed) {
            return None;
        }
        if let Some(winner) = E::G::get_winner(s) {
            return Some(winner.evaluate());
        }
        if depth == 0 {
            return Some(self.eval.evaluate(s));
        }

        let mut moves = Vec::new();
        self.move_pool.local_do(|pool| moves = pool.alloc());
        self.eval.generate_noisy_moves(s, &mut moves);
        if moves.is_empty() {
            self.move_pool.local_do(|pool| pool.free(moves));
            return Some(self.eval.evaluate(s));
        }

        let mut best = WORST_EVAL;
        for m in moves.iter() {
            m.apply(s);
            let value = -self.noisy_negamax(s, depth - 1, -beta, -alpha)?;
            m.undo(s);
            best = max(best, value);
            alpha = max(alpha, value);
            if alpha >= beta {
                break;
            }
        }
        self.move_pool.local_do(|pool| pool.free(moves));
        Some(best)
    }

    // Recursively compute negamax on the game state. Returns None if it hits the timeout.
    fn negamax(
        &self, s: &mut <E::G as Game>::S, prev_move: Option<<E::G as Game>::M>, depth: u8,
        mut alpha: Evaluation, mut beta: Evaluation,
    ) -> Option<Evaluation>
    where
        <E::G as Game>::S: Clone + Zobrist + Send + Sync,
        <E::G as Game>::M: Copy + Eq + Send + Sync,
        E: Sync,
    {
        if self.timeout.load(Ordering::Relaxed) {
            return None;
        }

        self.stats.local_do(|stats| stats.explore_node());

        if depth == 0 {
            // Evaluate quiescence search on leaf nodes.
            // Will just return the node's evaluation if quiescence search is disabled.
            return self.noisy_negamax(s, self.opts.max_quiescence_depth, alpha, beta);
        }
        if let Some(winner) = E::G::get_winner(s) {
            return Some(winner.evaluate());
        }

        let alpha_orig = alpha;
        let hash = s.zobrist_hash();
        let mut good_move = None;
        if let Some(value) = self.table.check(hash, depth, &mut good_move, &mut alpha, &mut beta) {
            return Some(value);
        }

        if self.null_move_check(s, depth, beta)? >= beta {
            return Some(beta);
        }

        let mut moves = Vec::new();
        self.move_pool.local_do(|pool| moves = pool.alloc());
        E::G::generate_moves(s, &mut moves);
        self.stats.local_do(|stats| stats.generate_moves(moves.len()));
        if moves.is_empty() {
            self.move_pool.local_do(|pool| pool.free(moves));
            return Some(WORST_EVAL);
        }

        // Reorder moves.
        if depth >= self.opts.min_reorder_moves_depth {
            self.eval.reorder_moves(s, &mut moves);
        }
        self.countermoves.local_do(|cm| cm.reorder(prev_move, &mut moves));
        if let Some(good) = good_move {
            move_to_front(good, &mut moves);
        }

        let first_move = moves[0];

        // Evaluate first move serially.
        first_move.apply(s);
        let initial_value = -self.negamax(s, Some(first_move), depth - 1, -beta, -alpha)?;
        first_move.undo(s);
        alpha = max(alpha, initial_value);
        let (best, best_move) = if alpha >= beta {
            // Skip search
            (initial_value, first_move)
        } else if self.par_opts.serial_cutoff_depth >= depth {
            // Serial search
            let mut best = initial_value;
            let mut best_move = first_move;
            let mut null_window = false;
            for &m in moves[1..].iter() {
                m.apply(s);
                let value = if null_window {
                    let probe = -self.negamax(s, Some(m), depth - 1, -alpha - 1, -alpha)?;
                    if probe > alpha && probe < beta {
                        // Full search fallback.
                        -self.negamax(s, Some(m), depth - 1, -beta, -probe)?
                    } else {
                        probe
                    }
                } else {
                    -self.negamax(s, Some(m), depth - 1, -beta, -alpha)?
                };
                m.undo(s);
                if value > best {
                    best = value;
                    best_move = m;
                }
                if value > alpha {
                    alpha = value;
                    // Now that we've found a good move, assume following moves
                    // are worse, and seek to cull them without full evaluation.
                    null_window = self.opts.null_window_search;
                }
                if alpha >= beta {
                    self.countermoves.local_do(|cm| cm.update(prev_move, m));
                    break;
                }
            }
            (best, best_move)
        } else {
            let alpha = AtomicI16::new(alpha);
            let best_move = Mutex::new(ValueMove::new(initial_value, first_move));
            // Parallel search
            let result = moves[1..].par_iter().with_max_len(1).try_for_each(|&m| -> Option<()> {
                // Check to see if we're cancelled by another branch.
                let initial_alpha = alpha.load(Ordering::SeqCst);
                if initial_alpha >= beta {
                    return None;
                }

                let mut state = s.clone();
                m.apply(&mut state);
                let value = if self.opts.null_window_search && initial_alpha > alpha_orig {
                    // TODO: send reference to alpha as neg_beta to children.
                    let probe = -self.negamax(
                        &mut state,
                        Some(m),
                        depth - 1,
                        -initial_alpha - 1,
                        -initial_alpha,
                    )?;
                    if probe > initial_alpha && probe < beta {
                        // Check again that we're not cancelled.
                        if alpha.load(Ordering::SeqCst) >= beta {
                            return None;
                        }
                        // Full search fallback.
                        -self.negamax(&mut state, Some(m), depth - 1, -beta, -probe)?
                    } else {
                        probe
                    }
                } else {
                    -self.negamax(&mut state, Some(m), depth - 1, -beta, -initial_alpha)?
                };

                alpha.fetch_max(value, Ordering::SeqCst);
                let mut bests = best_move.lock().unwrap();
                bests.max(value, m);
                Some(())
            });
            if result.is_none() {
                // Check for timeout.
                if self.timeout.load(Ordering::Relaxed) {
                    return None;
                }
            }
            best_move.into_inner().unwrap().into_inner()
        };

        self.table.concurrent_update(hash, alpha_orig, beta, depth, best, best_move);
        self.move_pool.local_do(|pool| pool.free(moves));
        Some(clamp_value(best))
    }

    fn iterative_search(
        &self, mut state: <E::G as Game>::S, max_depth: u8, background: bool,
    ) -> Option<(<E::G as Game>::M, Evaluation)> {
        self.table.concurrent_advance_generation();
        let root_hash = state.zobrist_hash();
        let mut best_move = None;
        let mut best_value = 0;
        let mut interval_start;
        let mut pv = String::new();

        let mut depth = max_depth % self.opts.step_increment;
        if depth == 0 {
            depth = self.opts.step_increment;
        }
        while depth <= max_depth as u8 {
            interval_start = Instant::now();
            if self.negamax(&mut state, None, depth, WORST_EVAL, BEST_EVAL).is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }
            let entry = match self.table.lookup(root_hash) {
                Some(entry) => entry,
                None => {
                    if background {
                        // Main tasks overwrote our result, just bail early.
                        return None;
                    } else {
                        panic!("Probably some race condition ate the best entry.");
                    }
                }
            };

            best_move = entry.best_move;
            best_value = entry.value;

            if self.opts.verbose && !background {
                let interval = Instant::now() - interval_start;
                eprintln!(
                    "Parallel (threads={}) depth{:>2} took{:>5}ms; returned{:>5}; bestmove {}",
                    self.par_opts.num_threads(),
                    depth,
                    interval.as_millis(),
                    entry.value_string(),
                    move_id::<E::G>(&mut state, best_move)
                );
            }

            depth += self.opts.step_increment;
            let mut pv_moves = Vec::new();
            self.table.populate_pv(&mut pv_moves, &mut state);
            self.pv.lock().unwrap().clone_from(&pv_moves);
            pv = pv_string::<E::G>(&pv_moves[..], &mut state);
            if unclamp_value(entry.value).abs() == BEST_EVAL {
                break;
            }
        }
        if self.opts.verbose && !background {
            eprintln!("Principal variation: {}", pv);
        }
        best_move.map(|m| (m, best_value))
    }
}

fn pretty_stats(stats: &Stats, start: Instant) -> String {
    let mean_branching_factor =
        stats.total_generated_moves as f64 / stats.total_generate_move_calls as f64;
    let throughput = (stats.nodes_explored) as f64 / (Instant::now() - start).as_secs_f64();
    format!(
        "Explored {} nodes. MBF={:.1}\n{} nodes/sec",
        stats.nodes_explored, mean_branching_factor, throughput as usize
    )
}

pub struct ParallelSearch<E: Evaluator> {
    max_depth: u8,
    max_time: Duration,

    background_cancel: Arc<AtomicBool>,
    table: Arc<LockfreeTable<<E::G as Game>::M>>,
    prev_value: Evaluation,
    principal_variation: Vec<<E::G as Game>::M>,
    eval: E,

    thread_pool: rayon::ThreadPool,

    opts: IterativeOptions,
    par_opts: ParallelOptions,
}

impl<E: Evaluator> ParallelSearch<E> {
    pub fn new(eval: E, opts: IterativeOptions, par_opts: ParallelOptions) -> ParallelSearch<E> {
        let table = Arc::new(LockfreeTable::new(opts.table_byte_size));
        let num_threads = par_opts.num_threads();
        let pool_builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);
        ParallelSearch {
            max_depth: 99,
            max_time: Duration::from_secs(5),
            background_cancel: Arc::new(AtomicBool::new(false)),
            table,
            prev_value: 0,
            principal_variation: Vec::new(),
            thread_pool: pool_builder.build().unwrap(),
            opts,
            par_opts,
            eval,
        }
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }
}

impl<E: Evaluator> Strategy<E::G> for ParallelSearch<E>
where
    <E::G as Game>::S: Clone + Zobrist + Send + Sync,
    <E::G as Game>::M: Copy + Eq + Send + Sync,
    E: Clone + Sync + Send + 'static,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        // Cancel any ongoing background processing.
        self.background_cancel.store(true, Ordering::SeqCst);
        // Start timer if configured.
        let timeout = if self.max_time == Duration::new(0, 0) {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.max_time)
        };

        let (best_move, value) = {
            let start_time = Instant::now();
            let mut negamaxer = ParallelNegamaxer::new(
                self.opts,
                self.par_opts,
                self.eval.clone(),
                self.table.clone(),
                timeout,
                &self.thread_pool,
            );
            // Launch in threadpool and wait for result.
            let value_move = self
                .thread_pool
                .install(|| negamaxer.iterative_search(s.clone(), self.max_depth, false));
            self.principal_variation = negamaxer.principal_variation();
            let mut stats = Stats::default();
            negamaxer.stats.do_all(|local| stats.add(local));
            if self.opts.verbose {
                eprintln!("{}", pretty_stats(&stats, start_time));
            }
            value_move
        }?;
        self.prev_value = value;

        if self.par_opts.background_pondering {
            self.background_cancel = Arc::new(AtomicBool::new(false));
            // Create a separate negamaxer to have a dedicated cancel
            // signal, and to allow the negamaxer to outlive this scope.
            let negamaxer = ParallelNegamaxer::new(
                self.opts,
                self.par_opts,
                self.eval.clone(),
                self.table.clone(),
                self.background_cancel.clone(),
                &self.thread_pool,
            );
            let mut state = s.clone();
            best_move.apply(&mut state);
            // Launch in threadpool asynchronously.
            self.thread_pool.spawn(move || {
                negamaxer.iterative_search(state, 99, true);
            });
        }
        Some(best_move)
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
        self.principal_variation.clone()
    }
}

impl<E: Evaluator> Drop for ParallelSearch<E> {
    fn drop(&mut self) {
        self.background_cancel.store(true, Ordering::SeqCst);
    }
}
