extern crate parking_lot;

use crate::interface::*;
use parking_lot::Mutex;
use std::cmp::{max, min};
use std::sync::atomic::{AtomicU8, Ordering};

// Common transposition table stuff.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum EntryFlag {
    Exact,
    Upperbound,
    Lowerbound,
}

// TODO: Optimize size. Ideally 16 bytes or less.
#[derive(Copy, Clone)]
pub(super) struct Entry<M> {
    pub(super) hash: u64,
    pub(super) value: Evaluation,
    pub(super) depth: u8,
    pub(super) flag: EntryFlag,
    pub(super) generation: u8,
    pub(super) best_move: Option<M>,
}

#[test]
fn test_entry_size() {
    // TODO: ratchet down
    assert!(std::mem::size_of::<Entry<u32>>() <= 24);
    assert!(std::mem::size_of::<Mutex<Entry<u32>>>() <= 32);
}

// It would be nice to unify most of the implementation of the single-threaded
// and concurrent tables, but the methods need different signatures.
pub(super) struct ConcurrentTable<M> {
    table: Vec<Mutex<Entry<M>>>,
    mask: usize,
    // Incremented for each iterative deepening run.
    // Values from old generations are always overwritten.
    generation: AtomicU8,
}

impl<M> ConcurrentTable<M> {
    pub(super) fn new(table_byte_size: usize) -> Self {
        let size = (table_byte_size / std::mem::size_of::<Mutex<Entry<M>>>()).next_power_of_two();
        let mask = (size - 1) & !1;
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Mutex::new(Entry::<M> {
                hash: 0,
                value: 0,
                depth: 0,
                flag: EntryFlag::Exact,
                generation: 0,
                best_move: None,
            }));
        }
        Self { table, mask, generation: AtomicU8::new(0) }
    }

    pub(super) fn advance_generation(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl<M> ConcurrentTable<M>
where
    M: Copy,
{
    // Using two-tier table, look in the two adjacent slots
    pub(super) fn lookup(&self, hash: u64) -> Option<Entry<M>> {
        let index = (hash as usize) & self.mask;
        for i in index..index + 2 {
            let entry = self.table[i].lock();
            if hash == entry.hash {
                return Some(*entry);
            }
        }
        None
    }

    fn store(&self, hash: u64, value: Evaluation, depth: u8, flag: EntryFlag, best_move: M) {
        let table_gen = self.generation.load(Ordering::Relaxed);
        // index points to the first of a pair of entries, the depth-preferred entry and the always-replace entry.
        let index = (hash as usize) & self.mask;
        let new_entry =
            Entry { hash, value, depth, flag, generation: table_gen, best_move: Some(best_move) };
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

    // Check and update negamax state based on any transposition table hit.
    // Returns Some(value) on an exact match.
    // Returns None, updating mutable arguments, if Negamax should continue to explore this node.
    pub(super) fn check(
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
    pub(super) fn update(
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
        self.store(hash, best, depth, flag, best_move);
    }
}
