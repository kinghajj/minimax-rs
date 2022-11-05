# minimax-rs - Generic implementations of Minimax in Rust.

[![Build Status](https://api.travis-ci.com/edre/minimax-rs.svg?branch=master)](https://travis-ci.com/github/edre/minimax-rs)
[![Crates.io](https://img.shields.io/crates/v/minimax.svg)](https://crates.io/crates/minimax)
[![Documentation](https://docs.rs/minimax/badge.svg)](https://docs.rs/minimax)

## About

This library provides interfaces that describe:

1. the rules for two-player, perfect-knowledge games;
2. methods of evaluating particular game states for a player; and
3. strategies for choosing moves for a player.

This crate implements multiple different strategies, so that any combination of
custom evaluators and strategies can be tested against each other. These include
single- and multi-threaded algorithms using alpha-beta pruning, iterative
deepening, and transposition tables. There is also a basic implementation of
multi-threaded Monte Carlo Tree Search, which does not require writing an
evaluator.

## Example

The `ttt` and `connect4` modules contain implementations of Tic-Tac-Toe and
Connect Four, demonstrating how to use the game and evaluation interfaces.
`test` shows how to use strategies.
