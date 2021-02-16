//! An implementation of iterative search.
//!
//! Search and evaluate at depth 0, then start over at depth 1, then depth 2,
//! etc. Can keep going until a maximum depth or maximum time or either. Uses
//! a transposition table to reuse information from previous iterations.

use super::super::interface::*;

use std::cmp::{max, min};
use std::marker::PhantomData;
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

#[derive(Copy, Clone, Eq, PartialEq)]
enum EntryFlag {
    Exact,
    Upperbound,
    Lowerbound,
}

// TODO: Optimize size.
#[derive(Copy, Clone)]
struct Entry<M> {
    hash: u64,
    value: Evaluation,
    depth: u8,
    flag: EntryFlag,
    best_move: Option<M>,
}

struct TranspositionTable<M> {
    table: Vec<Entry<M>>,
    mask: usize,
    minimum_depth: u8,
}

impl<M> TranspositionTable<M> {
    fn new(table_byte_size: usize) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Entry<M>>()).next_power_of_two();
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Entry::<M> {
                hash: 0,
                value: 0,
                depth: 0,
                flag: EntryFlag::Exact,
                best_move: None,
            });
        }
        Self { table: table, mask: size - 1, minimum_depth: 1 }
    }

    fn lookup(&self, hash: u64) -> Option<&Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        if hash == entry.hash {
            Some(entry)
        } else {
            None
        }
    }

    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        if depth >= self.minimum_depth {
            let index = (hash as usize) & self.mask;
            self.table[index] =
		Entry { hash: hash, value: value, depth: depth, flag: flag, best_move: Some(best_move) }
	}
    }
}

/// Options to use for the iterative search engine.
#[derive(Clone, Copy)]
pub struct IterativeOptions {
    table_byte_size: usize,
    // TODO: support more configuration of replacement strategy
    // https://www.chessprogramming.org/Transposition_Table#Replacement_Strategies
}

impl IterativeOptions {
    pub fn new() -> Self {
        IterativeOptions { table_byte_size: 1_000_000 }
    }
}

impl IterativeOptions {
    pub fn with_table_byte_size(mut self, size: usize) -> Self {
        self.table_byte_size = size;
        self
    }
}

pub struct IterativeSearch<E: Evaluator> {
    max_depth: usize,
    max_time: Duration,
    timeout: Arc<AtomicBool>,
    transposition_table: TranspositionTable<<<E as Evaluator>::G as Game>::M>,
    prev_value: Evaluation,
    _eval: PhantomData<E>,

    // Runtime stats for the last move generated.

    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored up to this depth.
    nodes_explored: usize,
    // Nodes explored past this depth, and thus this is thrown away work.
    next_depth_nodes: usize,
    table_hits: usize,
    wall_time: Duration,
}

impl<E: Evaluator> IterativeSearch<E> {
    pub fn new(opts: IterativeOptions) -> IterativeSearch<E> {
        let table = TranspositionTable::new(opts.table_byte_size);
        IterativeSearch {
            max_depth: 100,
            max_time: Duration::from_secs(5),
            timeout: Arc::new(AtomicBool::new(false)),
            transposition_table: table,
            prev_value: 0,
            _eval: PhantomData,
            actual_depth: 0,
            nodes_explored: 0,
            next_depth_nodes: 0,
            table_hits: 0,
            wall_time: Duration::default(),
        }
    }

    /// Set the maximum depth to search. Disables the timeout.
    /// This can be changed between moves while reusing the transposition table.
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
        self.max_time = Duration::new(0, 0);
    }

    /// The maximum time to compute the best move. When the timeout is hit, it
    /// returns the best move found of the previous full iteration. Unlimited
    /// max depth.
    pub fn set_timeout(&mut self, max_time: Duration) {
        self.max_time = max_time;
        self.max_depth = 100;
    }

    /// Return a human-readable summary of the last move generation.
    pub fn stats(&self) -> String {
        let throughput =
            (self.nodes_explored + self.next_depth_nodes) as f64 / self.wall_time.as_secs_f64();
        format!("Explored {} nodes to depth {}.\nInterrupted exploration of next depth explored {} nodes.\n{} transposition table hits.\n{} nodes/sec",
		self.nodes_explored, self.actual_depth, self.next_depth_nodes, self.table_hits, throughput as usize)
    }

    /// Return the value computed for the root node for the last computation.
    pub fn root_value(&self) -> Evaluation {
        self.prev_value
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

        if let Some(winner) = E::G::get_winner(s) {
            return Some(winner.evaluate());
        }
        if depth == 0 {
            return Some(E::evaluate(s));
        }

        let alpha_orig = alpha;
        let hash = s.zobrist_hash();
        let mut good_move = None;
        if let Some(entry) = self.transposition_table.lookup(hash) {
            good_move = entry.best_move;
            self.table_hits += 1;
            if entry.depth >= depth {
                match entry.flag {
                    EntryFlag::Exact => {
                        return Some(entry.value);
                    }
                    EntryFlag::Lowerbound => {
                        alpha = max(alpha, entry.value);
                    }
                    EntryFlag::Upperbound => {
                        beta = min(beta, entry.value);
                    }
                }
                if alpha >= beta {
                    return Some(entry.value);
                }
            }
        }

        let mut moves = [None; 200];
        let n = E::G::generate_moves(s, &mut moves);
        // Rearrange so predicted good move is first.
        for i in 0..n {
            if moves[i] == good_move {
                moves.swap(0, i);
                break;
            }
        }

        let mut best = WORST_EVAL;
        let mut best_move = moves[0].unwrap();
        for m in moves.iter().take_while(|om| om.is_some()).map(|om| om.unwrap()) {
            m.apply(s);
            let value = -self.negamax(s, depth - 1, -beta, -alpha)?;
            m.undo(s);
            if value > best {
                best = value;
                best_move = m;
            }
            alpha = max(alpha, value);
            if alpha >= beta {
                break;
            }
        }

        let flag = if best <= alpha_orig {
            EntryFlag::Upperbound
        } else if best >= beta {
            EntryFlag::Lowerbound
        } else {
            EntryFlag::Exact
        };
        self.transposition_table.store(hash, best, depth, flag, best_move);

        Some(best)
    }
}

impl<E: Evaluator> Strategy<E::G> for IterativeSearch<E>
where
    <E::G as Game>::S: Clone + Zobrist,
    <E::G as Game>::M: Copy + Eq,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        // Reset stats.
        self.nodes_explored = 0;
        self.next_depth_nodes = 0;
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

        for depth in 0..=self.max_depth as u8 {
            if self.negamax(&mut s_clone, depth + 1, WORST_EVAL, BEST_EVAL).is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }
            let entry = self.transposition_table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            self.actual_depth = max(self.actual_depth, depth);
            self.nodes_explored += self.next_depth_nodes;
            self.prev_value = entry.value;
            self.next_depth_nodes = 0;
        }
        self.wall_time = start_time.elapsed();
        best_move
    }
}
