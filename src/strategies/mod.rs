//! Strategy implementations.

pub mod iterative;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod lazy_smp;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod mcts;
pub mod negamax;
pub mod random;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod ybw;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod sync_util;
mod table;
mod util;
