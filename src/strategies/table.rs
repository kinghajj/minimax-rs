extern crate parking_lot;

use crate::interface::*;
use parking_lot::Mutex;
use std::cmp::{max, min};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

// Common transposition table stuff.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum EntryFlag {
    Exact,
    Upperbound,
    Lowerbound,
}

#[derive(Copy, Clone)]
pub(super) struct Entry<M: Copy> {
    // High bits of hash. Low bits are used in table index.
    pub(super) hash: u32,
    pub(super) value: Evaluation,
    pub(super) depth: u8,
    pub(super) flag: EntryFlag,
    pub(super) generation: u8,
    // Always initialized when entry is populated.
    pub(super) best_move: MaybeUninit<M>,
}

#[test]
fn test_entry_size() {
    // Even with the mutex, we want this to be within 16 bytes so 4 can fit on a cache line.
    // 3 byte move allows enum Moves with 2 bytes of payload.
    assert!(std::mem::size_of::<Entry<[u8; 3]>>() <= 16);
    assert!(std::mem::size_of::<Mutex<Entry<[u8; 3]>>>() <= 16);
}

impl<M: Copy> Entry<M> {
    pub(super) fn empty() -> Self {
        Entry {
            hash: 0,
            value: 0,
            depth: 0,
            flag: EntryFlag::Exact,
            generation: 0,
            best_move: MaybeUninit::uninit(),
        }
    }

    pub(super) fn best_move(&self) -> M {
        debug_assert!(self.hash != 0);
        unsafe { self.best_move.assume_init() }
    }
}

pub(super) fn high_hash_bits(hash: u64) -> u32 {
    // Always set the bottom bit to ensure no one matches the zero hash.
    (hash >> 32) as u32 | 1
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
            *good_move = Some(entry.best_move());
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
            let m = entry.best_move();
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

// It would be nice to unify most of the implementation of the single-threaded
// and concurrent tables, but the methods need different signatures.
pub(super) struct ConcurrentTable<M: Copy> {
    table: Vec<Mutex<Entry<M>>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: AtomicU8,
}

impl<M: Copy> ConcurrentTable<M> {
    pub(super) fn new(table_byte_size: usize) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Mutex<Entry<M>>>()).next_power_of_two();
        let mask = (size - 1) & !1;
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Mutex::new(Entry::empty()));
        }
        Self { table, mask, generation: AtomicU8::new(0) }
    }

    pub(super) fn concurrent_advance_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl<M: Copy> Table<M> for ConcurrentTable<M> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        self.concurrent_lookup(hash)
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

impl<M: Copy> Table<M> for Arc<ConcurrentTable<M>> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        self.concurrent_lookup(hash)
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

impl<M> ConcurrentTable<M>
where
    M: Copy,
{
    // Using two-tier table, look in the two adjacent slots
    pub(super) fn concurrent_lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        for i in index..index + 2 {
            let entry = self.table[i].lock();
            if high_hash_bits(hash) == entry.hash {
                return Some(*entry);
            }
        }
        None
    }

    fn concurrent_store(
        &self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M,
    ) {
        let table_gen = self.generation.load(Ordering::Relaxed);
        // index points to the first of a pair of entries, the depth-preferred entry and the always-replace entry.
        let index = (hash as usize) & self.mask;
        let new_entry = Entry {
            hash: high_hash_bits(hash),
            value,
            depth,
            flag,
            generation: table_gen,
            best_move: MaybeUninit::new(best_move),
        };
        {
            let mut entry = self.table[index].lock();
            if entry.generation != table_gen || entry.depth <= depth {
                *entry = new_entry;
                return;
            }
        }
        // Otherwise, always overwrite second entry.
        *self.table[index + 1].lock() = new_entry;
    }

    // Update table based on negamax results.
    pub(super) fn concurrent_update(
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

// A concurrent table that doesn't bother to use atomic operations to access its entries.
// It's crazily unsafe, but somehow StockFish gets away with this?
pub(super) struct RacyTable<M: Copy> {
    table: Vec<Entry<M>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: AtomicU8,
}

#[allow(dead_code)]
impl<M: Copy> RacyTable<M> {
    pub(super) fn new(table_byte_size: usize) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Entry<M>>()).next_power_of_two();
        let mask = size - 1;
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Entry::empty());
        }
        Self { table, mask, generation: AtomicU8::new(0) }
    }

    pub(super) fn concurrent_advance_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl<M: Copy> Table<M> for RacyTable<M> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        self.concurrent_lookup(hash)
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

impl<M: Copy> Table<M> for Arc<RacyTable<M>> {
    fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        self.concurrent_lookup(hash)
    }
    fn store(&mut self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        self.concurrent_store(hash, value, depth, flag, best_move)
    }
    fn advance_generation(&mut self) {
        self.concurrent_advance_generation()
    }
}

#[allow(dead_code)]
impl<M> RacyTable<M>
where
    M: Copy,
{
    pub(super) fn concurrent_lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        let entry = self.table[index];
        if high_hash_bits(hash) == entry.hash {
            return Some(entry);
        }
        None
    }

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
                hash: high_hash_bits(hash),
                value,
                depth,
                flag,
                generation: table_gen,
                best_move: MaybeUninit::new(best_move),
            };
        }
    }

    // Update table based on negamax results.
    pub(super) fn concurrent_update(
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
