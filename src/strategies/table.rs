use crate::interface::*;
use std::cmp::{max, min};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

// Common transposition table stuff.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum EntryFlag {
    Exact,
    Upperbound,
    Lowerbound,
}

#[derive(Copy, Clone)]
pub(super) struct Entry<M> {
    pub(super) high_hash: u32,
    pub(super) value: Evaluation,
    pub(super) depth: u8,
    pub(super) flag: EntryFlag,
    pub(super) generation: u8,
    pub(super) best_move: Option<M>,
}

#[test]
fn test_entry_size() {
    assert!(std::mem::size_of::<Entry<[u8; 4]>>() <= 16);
    assert!(std::mem::size_of::<ConcurrentEntry<[u8; 4]>>() <= 16);
}

pub(super) fn high_bits(hash: u64) -> u32 {
    (hash >> 32) as u32
}

// A trait for a transposition table. The methods are mutual exclusion, but
// the idea is that an implementation can wrap a shared concurrent table.
pub(super) trait Table<M: Copy> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>>;
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M);
    fn advance_generation(&mut self);

    // Check and update negamax state based on any transposition table hit.
    // Returns Some(value) on an exact match.
    // Returns None, updating mutable arguments, if Negamax should continue to explore this node.
    fn check(
        &self, hash: u64, depth: u8, good_move: &mut Option<M>, alpha: &mut Evaluation,
        beta: &mut Evaluation,
    ) -> Option<Evaluation> {
        if let Some(entry) = self.lookup(hash) {
            *good_move = entry.best_move;
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
    fn update(
        &mut self, hash: u64, alpha_orig: Evaluation, beta: Evaluation, depth: u8,
        best: Evaluation, best_move: M,
    ) {
        let flag = if best <= alpha_orig {
            EntryFlag::Upperbound
        } else if best >= beta {
            EntryFlag::Lowerbound
        } else {
            EntryFlag::Exact
        };
        self.store(hash, best, depth, flag, best_move);
    }

    // After finishing a search, populate the principal variation as deep as
    // the table remembers it.
    fn populate_pv<G: Game>(&self, pv: &mut Vec<M>, s: &mut G::S, mut depth: u8)
    where
        M: Move<G = G>,
        <G as Game>::S: Zobrist,
    {
        pv.clear();
        let mut hash = s.zobrist_hash();
        while let Some(entry) = self.lookup(hash) {
            // The principal variation should only have exact nodes, as other
            // node types are from cutoffs where the node is proven to be
            // worse than a previously explored one.
            //
            // Sometimes, it takes multiple rounds of narrowing bounds for the
            // value to be exact, and we can't guarantee that the table entry
            // will remain in the table between the searches that find
            // equivalent upper and lower bounds.
            let m = entry.best_move.unwrap();
            pv.push(m);
            m.apply(s);
            hash = s.zobrist_hash();
            // Prevent cyclical PVs from being infinitely long.
            if depth == 0 {
                break;
            }
            depth -= 1;
        }
        // Restore state.
        for m in pv.iter().rev() {
            m.undo(s);
        }
    }
}

pub(super) trait ConcurrentTable<M> {
    fn concurrent_store(
        &self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M,
    );
    fn concurrent_advance_generation(&self);

    // Update table based on negamax results.
    fn concurrent_update(
        &self, hash: u64, alpha_orig: Evaluation, beta: Evaluation, depth: u8, best: Evaluation,
        best_move: M,
    ) {
        let flag = if best <= alpha_orig {
            EntryFlag::Upperbound
        } else if best >= beta {
            EntryFlag::Lowerbound
        } else {
            EntryFlag::Exact
        };
        self.concurrent_store(hash, best, depth, flag, best_move);
    }
}

impl<M: Copy, T: Table<M> + ConcurrentTable<M>> Table<M> for Arc<T> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        (**self).lookup(hash)
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

// A concurrent table that doesn't bother to use atomic operations to access its entries.
// It's crazily unsafe, but somehow StockFish gets away with this?
pub(super) struct RacyTable<M> {
    table: Vec<Entry<M>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: AtomicU8,
}

#[allow(dead_code)]
impl<M> RacyTable<M> {
    pub(super) fn new(table_byte_size: usize) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Entry<M>>()).next_power_of_two();
        let mask = size - 1;
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
        Self { table, mask, generation: AtomicU8::new(0) }
    }
}

impl<M: Copy> Table<M> for RacyTable<M> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = self.table[index];
        if high_bits(hash) == entry.high_hash {
            return Some(entry);
        }
        None
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

impl<M: Copy> ConcurrentTable<M> for RacyTable<M> {
    fn concurrent_store(
        &self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M,
    ) {
        let table_gen = self.generation.load(Ordering::Relaxed);
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        if entry.generation != table_gen || entry.depth <= depth {
            #[allow(mutable_transmutes)]
            let ptr = unsafe { std::mem::transmute::<&Entry<M>, &mut Entry<M>>(entry) };
            *ptr = Entry {
                high_hash: high_bits(hash),
                value,
                depth,
                flag,
                generation: table_gen,
                best_move: Some(best_move),
            };
        }
    }

    fn concurrent_advance_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

struct ConcurrentEntry<M> {
    high_hash: AtomicU32,
    value: Evaluation,
    depth: u8,
    flag: EntryFlag,
    generation: u8,
    best_move: Option<M>,
}

pub(super) struct LockfreeTable<M> {
    table: Vec<ConcurrentEntry<M>>,
    mask: usize,
    generation: AtomicU8,
}

// Safe for cross-thread usage because of manual concurrency operations.
unsafe impl<M> Sync for LockfreeTable<M> {}

impl<M: Copy> Table<M> for LockfreeTable<M> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        let table_hash = entry.high_hash.load(Ordering::SeqCst);
        if high_bits(hash) | 1 == table_hash | 1 {
            // Copy contents
            let ret = Some(Entry {
                // No one reads the hash.
                high_hash: 0,
                value: entry.value,
                depth: entry.depth,
                flag: entry.flag,
                generation: entry.generation,
                best_move: entry.best_move,
            });
            // Verify the hash hasn't changed during the copy.
            if table_hash == entry.high_hash.load(Ordering::SeqCst) {
                return ret;
            }
        }
        None
    }

    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

#[allow(dead_code)]
impl<M> LockfreeTable<M> {
    const WRITING_SENTINEL: u32 = 0xffff_ffff;

    pub(super) fn new(table_byte_size: usize) -> Self {
        let size =
            (table_byte_size / std::mem::size_of::<ConcurrentEntry<M>>()).next_power_of_two();
        let mask = size - 1;
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(ConcurrentEntry::<M> {
                high_hash: AtomicU32::new(0x5555_5555),
                value: 0,
                depth: 0,
                flag: EntryFlag::Exact,
                generation: 0,
                best_move: None,
            });
        }
        Self { table, mask, generation: AtomicU8::new(0) }
    }
}

impl<M: Copy> ConcurrentTable<M> for LockfreeTable<M> {
    fn concurrent_store(
        &self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M,
    ) {
        let table_gen = self.generation.load(Ordering::Relaxed);
        let index = (hash as usize) & self.mask;
        let entry = &self.table[index];
        if entry.generation != table_gen || entry.depth <= depth {
            // Set hash to sentinel value during write.
            let x = entry.high_hash.load(Ordering::SeqCst);
            if x == Self::WRITING_SENTINEL {
                // Someone's already writing, just forget it.
                return;
            }
            // Try to set to sentinel value:
            if entry
                .high_hash
                .compare_exchange_weak(
                    x,
                    Self::WRITING_SENTINEL,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                // Someone just started writing, just forget it.
                return;
            }

            // concurrent_lookup will throw out any read that occurs across a write.
            #[allow(mutable_transmutes)]
            let entry = unsafe {
                std::mem::transmute::<&ConcurrentEntry<M>, &mut ConcurrentEntry<M>>(entry)
            };
            entry.value = value;
            entry.depth = depth;
            entry.flag = flag;
            entry.generation = table_gen;
            entry.best_move = Some(best_move);

            // Set hash to correct value to indicate done.
            let new_hash = if high_bits(hash) | 1 == x | 1 {
                // If we're overwriting the same hash, flip the lowest bit to
                // catch any readers reading across this change.
                x ^ 1
            } else {
                high_bits(hash)
            };
            entry.high_hash.store(new_hash, Ordering::SeqCst);
        }
    }

    fn concurrent_advance_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}
