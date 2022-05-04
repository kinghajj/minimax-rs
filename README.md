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

## License

    Copyright (c) 2015 Samuel Fredrickson

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
