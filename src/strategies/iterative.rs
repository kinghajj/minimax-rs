//! An implementation of iterative deepening evaluation.
//!
//! Search and evaluate at depth 0, then start over at depth 1, then depth 2,
//! etc. Can keep going until a maximum depth or maximum time or either. Uses
//! a transposition table to reuse information from previous iterations.

use super::super::interface::*;
use super::super::util::*;
use super::util::*;

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

fn timeout_signal(dur: Duration) -> Arc<AtomicBool> {
    // Theoretically we could include an async runtime to do this and use
    // fewer threads, but the stdlib implementation is only a few lines...
    let signal = Arc::new(AtomicBool::new(false));
    let signal2 = signal.clone();
    spawn(move || {
        sleep(dur);
        signal2.store(true, Ordering::Relaxed);
    });
    signal
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// Strategies for when to overwrite entries in the transition table.
pub enum Replacement {
    Always,
    DepthPreferred,
    TwoTier,
    // TODO: Bucket(size)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum EntryFlag {
    Exact,
    Upperbound,
    Lowerbound,
}

// TODO: Optimize size. Ideally 16 bytes or less.
#[derive(Copy, Clone)]
struct Entry<M> {
    hash: u64,
    value: Evaluation,
    depth: u8,
    flag: EntryFlag,
    generation: u8,
    best_move: Option<M>,
}

struct TranspositionTable<M> {
    table: Vec<Entry<M>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: u8,
    strategy: Replacement,
}

impl<M> TranspositionTable<M> {
    fn new(table_byte_size: usize, strategy: Replacement) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Entry<M>>()).next_power_of_two();
        let mask = if strategy == Replacement::TwoTier { (size - 1) & !1 } else { size - 1 };
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Entry::<M> {
                hash: 0,
                value: 0,
                depth: 0,
                flag: EntryFlag::Exact,
                generation: 0,
                best_move: None,
            });
        }
        Self { table, mask, generation: 0, strategy }
    }

    fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn lookup(&self, hash: u64) -> Option<&Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        if hash == entry.hash {
            Some(entry)
        } else if self.strategy == Replacement::TwoTier {
            let entry = &self.table[index + 1];
            if hash == entry.hash {
                Some(entry)
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
                hash,
                value,
                depth,
                flag,
                generation: self.generation,
                best_move: Some(best_move),
            }
        }
    }
}

/// Options to use for the iterative search engine.
#[derive(Clone, Copy)]
pub struct IterativeOptions {
    table_byte_size: usize,
    strategy: Replacement,
    null_window_search: bool,
    step_increment: u8,
    max_quiescence_depth: u8,
}

impl IterativeOptions {
    pub fn new() -> Self {
        IterativeOptions {
            table_byte_size: 1_000_000,
            strategy: Replacement::TwoTier,
            null_window_search: true,
            step_increment: 1,
            max_quiescence_depth: 0,
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
}

pub struct IterativeSearch<E: Evaluator> {
    max_depth: usize,
    max_time: Duration,
    timeout: Arc<AtomicBool>,
    transposition_table: TranspositionTable<<<E as Evaluator>::G as Game>::M>,
    move_pool: MovePool<<E::G as Game>::M>,
    prev_value: Evaluation,
    eval: E,

    opts: IterativeOptions,

    // Runtime stats for the last move generated.

    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored at each depth.
    nodes_explored: Vec<u64>,
    // Nodes explored past this depth, and thus only useful for filling TT for
    // next choose_move.
    next_depth_nodes: u64,
    // For computing the average branching factor.
    total_generate_move_calls: u64,
    total_generated_moves: u64,
    table_hits: usize,
    pv: Vec<<E::G as Game>::M>,
    wall_time: Duration,
}

impl<E: Evaluator> IterativeSearch<E> {
    pub fn new(eval: E, opts: IterativeOptions) -> IterativeSearch<E> {
        let table = TranspositionTable::new(opts.table_byte_size, opts.strategy);
        IterativeSearch {
            max_depth: 100,
            max_time: Duration::from_secs(5),
            timeout: Arc::new(AtomicBool::new(false)),
            transposition_table: table,
            move_pool: MovePool::<_>::default(),
            prev_value: 0,
            opts,
            eval,
            actual_depth: 0,
            nodes_explored: Vec::new(),
            next_depth_nodes: 0,
            total_generate_move_calls: 0,
            total_generated_moves: 0,
            table_hits: 0,
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
        let mean_branching_factor =
            self.total_generated_moves as f64 / self.total_generate_move_calls as f64;
        let effective_branching_factor = (*self.nodes_explored.last().unwrap_or(&0) as f64)
            .powf((self.actual_depth as f64 + 1.0).recip());
        let throughput =
            (total_nodes_explored + self.next_depth_nodes) as f64 / self.wall_time.as_secs_f64();
        format!("Explored {} nodes to depth {}. MBF={:.1} EBF={:.1}\nPartial exploration of next depth hit {} nodes.\n{} transposition table hits.\n{} nodes/sec",
		total_nodes_explored, self.actual_depth, mean_branching_factor, effective_branching_factor,
		self.next_depth_nodes, self.table_hits, throughput as usize)
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }

    // After finishing a search, populate the principal variation as deep as
    // the table remembers it.
    fn populate_pv(&mut self, s: &mut <E::G as Game>::S, mut depth: u8)
    where
        <E::G as Game>::S: Zobrist,
        <E::G as Game>::M: Copy,
    {
        self.pv.clear();
        let mut hash = s.zobrist_hash();
        while let Some(entry) = self.transposition_table.lookup(hash) {
            // The principal variation should only have exact nodes, as other
            // node types are from cutoffs where the node is proven to be
            // worse than a previously explored one.
            //
            // Sometimes, it takes multiple rounds of narrowing bounds for the
            // value to be exact, and we can't guarantee that the table entry
            // will remain in the table between the searches that find
            // equivalent upper and lower bounds.
            let m = entry.best_move.unwrap();
            self.pv.push(m);
            m.apply(s);
            hash = s.zobrist_hash();
            // Prevent cyclical PVs from being infinitely long.
            if depth == 0 {
                break;
            }
            depth -= 1;
        }
        // Restore state.
        for m in self.pv.iter().rev() {
            m.undo(s);
        }
    }

    /// Return what the engine considered to be the best sequence of moves
    /// from both sides.
    pub fn principal_variation(&self) -> &[<E::G as Game>::M] {
        &self.pv[..]
    }

    // Negamax only among noisy moves.
    fn noisy_negamax(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, mut alpha: Evaluation, beta: Evaluation,
    ) -> Option<Evaluation>
    where
        <E::G as Game>::M: Copy,
    {
        if self.timeout.load(Ordering::Relaxed) {
            return None;
        }
        if let Some(winner) = E::G::get_winner(s) {
            return Some(winner.evaluate());
        }
        if depth == 0 {
            return Some(self.eval.evaluate(s));
        }

        let mut moves = self.move_pool.alloc();
        E::G::generate_noisy_moves(s, &mut moves);
        if moves.is_empty() {
            // Only quiet moves remain, return leaf evaluation.
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

    // Check and update negamax state based on any transposition table hit.
    #[inline]
    fn table_check(
        &mut self, hash: u64, depth: u8, good_move: &mut Option<<E::G as Game>::M>,
        alpha: &mut Evaluation, beta: &mut Evaluation,
    ) -> Option<Evaluation>
    where
        <E::G as Game>::M: Copy,
    {
        if let Some(entry) = self.transposition_table.lookup(hash) {
            *good_move = entry.best_move;
            self.table_hits += 1;
            if entry.depth >= depth {
                match entry.flag {
                    EntryFlag::Exact => {
                        return Some(entry.value);
                    }
                    EntryFlag::Lowerbound => {
                        *alpha = max(*alpha, entry.value);
                    }
                    EntryFlag::Upperbound => {
                        *beta = min(*beta, entry.value);
                    }
                }
                if *alpha >= *beta {
                    return Some(entry.value);
                }
            }
        }
        None
    }

    // Update table based on negamax results.
    #[inline(always)]
    fn table_update(
        &mut self, hash: u64, alpha_orig: Evaluation, beta: Evaluation, depth: u8,
        best: Evaluation, best_move: <E::G as Game>::M,
    ) {
        let flag = if best <= alpha_orig {
            EntryFlag::Upperbound
        } else if best >= beta {
            EntryFlag::Lowerbound
        } else {
            EntryFlag::Exact
        };
        self.transposition_table.store(hash, best, depth, flag, best_move);
    }

    // Recursively compute negamax on the game state. Returns None if it hits the timeout.
    fn negamax(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, mut alpha: Evaluation,
        mut beta: Evaluation,
    ) -> Option<Evaluation>
    where
        <E::G as Game>::S: Zobrist,
        <E::G as Game>::M: Copy + Eq,
    {
        if self.timeout.load(Ordering::Relaxed) {
            return None;
        }

        self.next_depth_nodes += 1;

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
        if let Some(value) = self.table_check(hash, depth, &mut good_move, &mut alpha, &mut beta) {
            return Some(value);
        }

        let mut moves = self.move_pool.alloc();
        E::G::generate_moves(s, &mut moves);
        self.total_generate_move_calls += 1;
        self.total_generated_moves += moves.len() as u64;
        if moves.is_empty() {
            self.move_pool.free(moves);
            return Some(WORST_EVAL);
        }
        if let Some(good) = good_move {
            // Rearrange so predicted good move is first.
            for i in 0..moves.len() {
                if moves[i] == good {
                    moves.swap(0, i);
                    break;
                }
            }
        }

        let mut best = WORST_EVAL;
        let mut best_move = moves[0];
        let mut null_window = false;
        for &m in moves.iter() {
            m.apply(s);
            let value = if null_window {
                let probe = -self.negamax(s, depth - 1, -alpha - 1, -alpha)?;
                if probe > alpha && probe < beta {
                    // Full search fallback.
                    -self.negamax(s, depth - 1, -beta, -probe)?
                } else {
                    probe
                }
            } else {
                -self.negamax(s, depth - 1, -beta, -alpha)?
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
                break;
            }
        }

        self.table_update(hash, alpha_orig, beta, depth, best, best_move);
        self.move_pool.free(moves);
        Some(clamp_value(best))
    }
}

impl<E: Evaluator> Strategy<E::G> for IterativeSearch<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        self.transposition_table.advance_generation();
        // Reset stats.
        self.nodes_explored.clear();
        self.next_depth_nodes = 0;
        self.total_generate_move_calls = 0;
        self.total_generated_moves = 0;
        self.actual_depth = 0;
        self.table_hits = 0;
        let start_time = Instant::now();
        // Start timer if configured.
        self.timeout = if self.max_time == Duration::new(0, 0) {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.max_time)
        };

        let root_hash = s.zobrist_hash();
        let mut s_clone = s.clone();
        let mut best_move = None;

        let mut depth = self.max_depth as u8 % self.opts.step_increment;
        while depth <= self.max_depth as u8 {
            if self.negamax(&mut s_clone, depth + 1, WORST_EVAL, BEST_EVAL).is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }
            let entry = self.transposition_table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            self.actual_depth = max(self.actual_depth, depth);
            self.nodes_explored.push(self.next_depth_nodes);
            self.prev_value = entry.value;
            self.next_depth_nodes = 0;
            depth += self.opts.step_increment;
            self.populate_pv(&mut s_clone, depth + 1);
        }
        self.wall_time = start_time.elapsed();
        best_move
    }
}
