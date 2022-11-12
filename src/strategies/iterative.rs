//! An implementation of iterative deepening evaluation.
//!
//! Search and evaluate at depth 1, then start over at depth 2, then depth 3,
//! etc. Can keep going until a maximum depth or maximum time or either. Uses
//! a transposition table to reuse information from previous iterations.

use super::super::interface::*;
use super::super::util::*;
#[cfg(not(target_arch = "wasm32"))]
use super::sync_util::timeout_signal;
use super::table::*;
use super::util::*;

use instant::Instant;
use rand::prelude::SliceRandom;
use std::cmp::max;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::time::Duration;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// Strategies for when to overwrite entries in the transition table.
pub enum Replacement {
    Always,
    DepthPreferred,
    TwoTier,
    // TODO: Bucket(size)
}

struct TranspositionTable<M> {
    table: Vec<Entry<M>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: u8,
    strategy: Replacement,
}

impl<M: Copy> TranspositionTable<M> {
    fn new(table_byte_size: usize, strategy: Replacement) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Entry<M>>()).next_power_of_two();
        let mask = if strategy == Replacement::TwoTier { (size - 1) & !1 } else { size - 1 };
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Entry::<M> {
                high_hash: 0,
                value: 0,
                depth: 0,
                flag: EntryFlag::Exact,
                generation: 0,
                best_move: None,
            });
        }
        Self { table, mask, generation: 0, strategy }
    }
}

impl<M: Copy> Table<M> for TranspositionTable<M> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        if high_bits(hash) == entry.high_hash {
            Some(*entry)
        } else if self.strategy == Replacement::TwoTier {
            let entry = &self.table[index + 1];
            if high_bits(hash) == entry.high_hash {
                Some(*entry)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        let dest = match self.strategy {
            Replacement::Always => Some((hash as usize) & self.mask),
            Replacement::DepthPreferred => {
                let index = (hash as usize) & self.mask;
                let entry = &self.table[index];
                if entry.generation != self.generation || entry.depth <= depth {
                    Some(index)
                } else {
                    None
                }
            }
            Replacement::TwoTier => {
                // index points to the first of a pair of entries, the depth-preferred entry and the always-replace entry.
                let index = (hash as usize) & self.mask;
                let entry = &self.table[index];
                if entry.generation != self.generation || entry.depth <= depth {
                    Some(index)
                } else {
                    Some(index + 1)
                }
            }
        };
        if let Some(index) = dest {
            self.table[index] = Entry {
                high_hash: high_bits(hash),
                value,
                depth,
                flag,
                generation: self.generation,
                best_move: Some(best_move),
            }
        }
    }

    fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
}

/// Options to use for the iterative search engines.
#[derive(Clone, Copy)]
pub struct IterativeOptions {
    pub table_byte_size: usize,
    pub(super) strategy: Replacement,
    pub(super) null_window_search: bool,
    pub(super) null_move_depth: Option<u8>,
    pub(super) singular_extension: bool,
    pub(super) aspiration_window: Option<Evaluation>,
    pub(super) mtdf: bool,
    pub(super) step_increment: u8,
    pub(super) max_quiescence_depth: u8,
    pub(super) min_reorder_moves_depth: u8,
    pub(super) countermove_table: bool,
    pub(super) countermove_history_table: bool,
    pub verbose: bool,
}

impl IterativeOptions {
    pub fn new() -> Self {
        IterativeOptions {
            table_byte_size: 1 << 20,
            strategy: Replacement::TwoTier,
            null_window_search: true,
            null_move_depth: None,
            singular_extension: false,
            aspiration_window: None,
            mtdf: false,
            step_increment: 1,
            max_quiescence_depth: 0,
            min_reorder_moves_depth: u8::MAX,
            countermove_table: false,
            countermove_history_table: false,
            verbose: false,
        }
    }
}

impl Default for IterativeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeOptions {
    /// Approximately how large the transposition table should be in memory.
    pub fn with_table_byte_size(mut self, size: usize) -> Self {
        self.table_byte_size = size;
        self
    }

    /// What rules to use when choosing whether to overwrite the current value
    /// in the transposition table.
    pub fn with_replacement_strategy(mut self, strategy: Replacement) -> Self {
        self.strategy = strategy;
        self
    }

    /// Whether to add null-window searches to try to prune branches that are
    /// probably worse than those already found. Also known as principal
    /// variation search.
    pub fn with_null_window_search(mut self, null: bool) -> Self {
        self.null_window_search = null;
        self
    }

    /// Whether to attempt to cut off early by seeing if each node is amazing
    /// even after passing the turn to the opponent. Null move search explores
    /// the tree at a depth reduced by this amount.
    pub fn with_null_move_depth(mut self, depth_reduction: u8) -> Self {
        self.null_move_depth = Some(depth_reduction);
        self
    }

    /// Whether to extend a branch of the search (by 1) if there is only one
    /// move (or only one reasonable move).
    pub fn with_singular_extension(mut self) -> Self {
        self.singular_extension = true;
        self
    }

    /// Whether to search first in a narrow window around the previous root
    /// value on each iteration.
    pub fn with_aspiration_window(mut self, window: Evaluation) -> Self {
        self.aspiration_window = Some(window);
        self
    }

    /// Whether to search for the correct value in each iteration using only
    /// null-window "Tests", with the
    /// [MTD(f)](https://en.wikipedia.org/wiki/MTD%28f%29) algorithm.
    /// Can be more efficient if the evaluation function is coarse grained.
    pub fn with_mtdf(mut self) -> Self {
        self.mtdf = true;
        self
    }

    /// Increment the depth by two between iterations.
    pub fn with_double_step_increment(mut self) -> Self {
        self.step_increment = 2;
        self
    }

    /// Enable [quiescence
    /// search](https://en.wikipedia.org/wiki/Quiescence_search) at the leaves
    /// of the search tree.  The Evaluator must implement `generate_noisy_moves`
    /// for the search to know when the state has become "quiet".
    pub fn with_quiescence_search_depth(mut self, depth: u8) -> Self {
        self.max_quiescence_depth = depth;
        self
    }

    /// Enable the Evaluator's move reordering after generating moves for all
    /// nodes at this depth or higher. Reordering can be an expensive
    /// operation, but it could cut off a lot of nodes if done well high in
    /// the search tree.
    pub fn with_min_reorder_moves_depth(mut self, depth: u8) -> Self {
        self.min_reorder_moves_depth = depth;
        self
    }

    /// Enable the countermove table, which reorders to the front moves that
    /// have worked to counter the previous move in other branches.
    pub fn with_countermoves(mut self) -> Self {
        self.countermove_table = true;
        self
    }

    /// Enable the countermove history table. It keeps a counter for moves
    /// that have caused beta cutoffs in other branches, and reorders moves
    /// based on this counter.
    pub fn with_countermove_history(mut self) -> Self {
        self.countermove_history_table = true;
        self
    }

    /// Enable verbose print statements of the ongoing performance of the search.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

#[derive(Default)]
pub(crate) struct Stats {
    pub(crate) nodes_explored: u64,
    pub(crate) total_generate_move_calls: u64,
    pub(crate) total_generated_moves: u64,
}

impl Stats {
    pub(crate) fn reset(&mut self) {
        self.nodes_explored = 0;
        self.total_generate_move_calls = 0;
        self.total_generated_moves = 0;
    }
    pub(crate) fn explore_node(&mut self) {
        self.nodes_explored += 1;
    }

    pub(crate) fn generate_moves(&mut self, num_moves: usize) {
        self.total_generate_move_calls += 1;
        self.total_generated_moves += num_moves as u64;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn add(&mut self, other: &Self) {
        self.nodes_explored += other.nodes_explored;
        self.total_generate_move_calls += other.total_generate_move_calls;
        self.total_generated_moves += other.total_generated_moves;
    }
}

pub(super) struct Negamaxer<E: Evaluator, T> {
    #[cfg(not(target_arch = "wasm32"))]
    timeout: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    deadline: Instant,
    #[cfg(target_arch = "wasm32")]
    timeout_counter: u32,
    pub(super) table: T,
    pub(super) countermoves: CounterMoves<<E::G as Game>::M>,
    move_pool: MovePool<<E::G as Game>::M>,
    eval: E,

    opts: IterativeOptions,
    pub(crate) stats: Stats,
}

impl<E: Evaluator, T: Table<<E::G as Game>::M>> Negamaxer<E, T>
where
    <E::G as Game>::S: Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    pub(super) fn new(table: T, eval: E, opts: IterativeOptions) -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            timeout: Arc::new(AtomicBool::new(false)),
            #[cfg(target_arch = "wasm32")]
            deadline: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            timeout_counter: 1000,
            table,
            countermoves: CounterMoves::new(opts.countermove_table, opts.countermove_history_table),
            eval,
            move_pool: MovePool::default(),
            opts,
            stats: Stats::default(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn set_timeout(&mut self, timeout: Arc<AtomicBool>) {
        self.timeout = timeout;
    }

    #[cfg(target_arch = "wasm32")]
    fn reset_timeout(&mut self, duration: Duration) {
        self.timeout_counter = if duration == Duration::new(0, 0) {
            // Too high counter that never hits the maximum.
            1000
        } else {
            0
        };
        self.deadline = Instant::now() + duration;
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn reset_timeout(&mut self, duration: Duration) {
        self.set_timeout(if duration == Duration::new(0, 0) {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(duration)
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn timeout_check(&mut self) -> bool {
        self.timeout_counter += 1;
        if self.timeout_counter != 100 {
            return false;
        }
        self.timeout_counter = 0;
        Instant::now() >= self.deadline
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn timeout_check(&mut self) -> bool {
        self.timeout.load(Ordering::Relaxed)
    }

    fn null_move_check(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, beta: Evaluation,
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
        &mut self, s: &mut <E::G as Game>::S, depth: u8, mut alpha: Evaluation, beta: Evaluation,
    ) -> Option<Evaluation> {
        if self.timeout_check() {
            return None;
        }
        if let Some(winner) = E::G::get_winner(s) {
            return Some(winner.evaluate());
        }
        if depth == 0 {
            return Some(self.eval.evaluate(s));
        }

        let mut moves = self.move_pool.alloc();
        self.eval.generate_noisy_moves(s, &mut moves);
        if moves.is_empty() {
            self.move_pool.free(moves);
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
        self.move_pool.free(moves);
        Some(best)
    }

    // Recursively compute negamax on the game state. Returns None if it hits the timeout.
    pub(super) fn negamax(
        &mut self, s: &mut <E::G as Game>::S, prev_move: Option<<E::G as Game>::M>, mut depth: u8,
        mut alpha: Evaluation, mut beta: Evaluation,
    ) -> Option<Evaluation> {
        if self.timeout_check() {
            return None;
        }

        self.stats.explore_node();

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

        let mut moves = self.move_pool.alloc();
        E::G::generate_moves(s, &mut moves);
        self.stats.generate_moves(moves.len());
        if moves.is_empty() {
            self.move_pool.free(moves);
            return Some(WORST_EVAL);
        }

        // TODO: Also do a pre-search to look for moves much better than others.
        if self.opts.singular_extension && moves.len() == 1 {
            depth += 1;
        }

        // Reorder moves.
        if depth >= self.opts.min_reorder_moves_depth {
            self.eval.reorder_moves(s, &mut moves);
        }
        self.countermoves.reorder(prev_move, &mut moves);
        if let Some(good) = good_move {
            move_to_front(good, &mut moves);
        }

        let mut best = WORST_EVAL;
        let mut best_move = moves[0];
        let mut null_window = false;
        for &m in moves.iter() {
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
                self.countermoves.update(prev_move, m);
                break;
            }
        }

        self.table.update(hash, alpha_orig, beta, depth, best, best_move);
        self.move_pool.free(moves);
        Some(clamp_value(best))
    }

    // Try to find the value within a window around the estimated value.
    // Results, whether exact, overshoot, or undershoot, are stored in the table.
    pub(super) fn aspiration_search(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, target: Evaluation, window: Evaluation,
    ) -> Option<()> {
        if depth < 2 {
            // Do a full search on shallow nodes to establish the target.
            return Some(());
        }
        let alpha = max(target.saturating_sub(window), WORST_EVAL);
        let beta = target.saturating_add(window);
        self.negamax(s, None, depth, alpha, beta)?;
        Some(())
    }

    pub(super) fn search_and_reorder(
        &mut self, s: &mut <E::G as Game>::S, moves: &mut [ValueMove<<E::G as Game>::M>], depth: u8,
    ) -> Option<Evaluation> {
        let mut alpha = WORST_EVAL;
        let beta = BEST_EVAL;
        for value_move in moves.iter_mut() {
            value_move.m.apply(s);
            let value = -self.negamax(s, Some(value_move.m), depth - 1, -beta, -alpha)?;
            value_move.m.undo(s);

            alpha = max(alpha, value);
            value_move.value = value;
        }
        moves.sort_by_key(|vm| -vm.value);
        self.table.update(s.zobrist_hash(), alpha, beta, depth, moves[0].value, moves[0].m);
        Some(moves[0].value)
    }
}

pub struct IterativeSearch<E: Evaluator> {
    max_depth: u8,
    max_time: Duration,
    negamaxer: Negamaxer<E, TranspositionTable<<E::G as Game>::M>>,
    prev_value: Evaluation,
    opts: IterativeOptions,

    // Runtime stats for the last move generated.

    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored at each depth.
    nodes_explored: Vec<u64>,
    pv: Vec<<E::G as Game>::M>,
    wall_time: Duration,
}

impl<E: Evaluator> IterativeSearch<E>
where
    <E::G as Game>::M: Copy + Eq,
    <E::G as Game>::S: Clone + Zobrist,
{
    pub fn new(eval: E, opts: IterativeOptions) -> IterativeSearch<E> {
        let table = TranspositionTable::new(opts.table_byte_size, opts.strategy);
        let negamaxer = Negamaxer::new(table, eval, opts);
        IterativeSearch {
            max_depth: 99,
            max_time: Duration::from_secs(5),
            prev_value: 0,
            negamaxer,
            opts,
            actual_depth: 0,
            nodes_explored: Vec::new(),
            pv: Vec::new(),
            wall_time: Duration::default(),
        }
    }

    /// Return a human-readable summary of the last move generation.
    pub fn stats(&self, s: &mut <E::G as Game>::S) -> String {
        let total_nodes_explored: u64 = self.nodes_explored.iter().sum();
        let mean_branching_factor = self.negamaxer.stats.total_generated_moves as f64
            / self.negamaxer.stats.total_generate_move_calls as f64;
        let effective_branching_factor = (*self.nodes_explored.last().unwrap_or(&0) as f64)
            .powf((self.actual_depth as f64 + 1.0).recip());
        let throughput = (total_nodes_explored + self.negamaxer.stats.nodes_explored) as f64
            / self.wall_time.as_secs_f64();
        format!("Principal variation: {}\nExplored {} nodes to depth {}. MBF={:.1} EBF={:.1}\nPartial exploration of next depth hit {} nodes.\n{} nodes/sec",
                pv_string::<E::G>(&self.pv[..], s),
		total_nodes_explored, self.actual_depth, mean_branching_factor, effective_branching_factor,
		self.negamaxer.stats.nodes_explored, throughput as usize)
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }

    fn mtdf(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, mut guess: Evaluation,
    ) -> Option<Evaluation> {
        let mut lowerbound = WORST_EVAL;
        let mut upperbound = BEST_EVAL;
        while lowerbound < upperbound {
            let beta = max(lowerbound + 1, guess);
            if self.opts.verbose {
                eprintln!(
                    "mtdf depth={} guess={} bounds={}:{}",
                    depth, beta, lowerbound, upperbound
                );
            }
            guess = self.negamaxer.negamax(s, None, depth, beta - 1, beta)?;
            if guess < beta {
                upperbound = guess;
            } else {
                lowerbound = guess;
            }
        }
        Some(guess)
    }
}

impl<E: Evaluator> Strategy<E::G> for IterativeSearch<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        self.negamaxer.table.advance_generation();
        self.negamaxer.countermoves.advance_generation(E::G::null_move(s));
        // Reset stats.
        self.nodes_explored.clear();
        self.negamaxer.stats.reset();
        self.actual_depth = 0;
        let start_time = Instant::now();
        // Start timer if configured.
        self.negamaxer.reset_timeout(self.max_time);

        let root_hash = s.zobrist_hash();
        let mut s_clone = s.clone();
        let mut best_move = None;
        let mut interval_start;
        // Store the moves so they can be reordered every iteration.
        let mut moves = Vec::new();
        E::G::generate_moves(&s_clone, &mut moves);
        // Start in a random order.
        moves.shuffle(&mut rand::thread_rng());
        let mut moves = moves.into_iter().map(|m| ValueMove::new(0, m)).collect::<Vec<_>>();

        // Start at 1 or 2 to hit the max depth.
        let mut depth = self.max_depth as u8 % self.opts.step_increment;
        if depth == 0 {
            depth = self.opts.step_increment;
        }
        while depth <= self.max_depth as u8 {
            interval_start = Instant::now();
            let search = if self.opts.mtdf {
                self.mtdf(&mut s_clone, depth, self.prev_value)
            } else {
                if let Some(window) = self.opts.aspiration_window {
                    // Results of the search are stored in the table.
                    if self
                        .negamaxer
                        .aspiration_search(&mut s_clone, depth, self.prev_value, window)
                        .is_none()
                    {
                        // Timeout.
                        break;
                    }
                    if self.opts.verbose {
                        if let Some(entry) = self.negamaxer.table.lookup(root_hash) {
                            let end = Instant::now();
                            let interval = end - interval_start;
                            eprintln!(
                                "Iterative aspiration depth{:>2} took{:>5}ms; bounds{:>5} bestmove={}",
                                depth,
                                interval.as_millis(),
                                entry.bounds(),
                                move_id::<E::G>(&mut s_clone, entry.best_move)
                            );
                            interval_start = end;
                        }
                    }
                }

                self.negamaxer.search_and_reorder(&mut s_clone, &mut moves[..], depth)
            };
            if search.is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }
            let entry = self.negamaxer.table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            if self.opts.verbose {
                let interval = Instant::now() - interval_start;
                eprintln!(
                    "Iterative fullsearch depth{:>2} took{:>5}ms; value{:>6} bestmove={}",
                    depth,
                    interval.as_millis(),
                    entry.value_string(),
                    move_id::<E::G>(&mut s_clone, best_move)
                );
            }

            self.actual_depth = max(self.actual_depth, depth);
            self.nodes_explored.push(self.negamaxer.stats.nodes_explored);
            self.negamaxer.stats.nodes_explored = 0;
            self.prev_value = entry.value;
            depth += self.opts.step_increment;
            self.negamaxer.table.populate_pv(&mut self.pv, &mut s_clone);
            if unclamp_value(entry.value).abs() == BEST_EVAL {
                break;
            }
        }
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
