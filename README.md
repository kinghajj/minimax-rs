# minimax-rs - Generic implementations of Minimax in Rust.

[![Build Status](https://travis-ci.org/kinghajj/minimax-rs.svg?branch=master)](https://travis-ci.org/kinghajj/minimax-rs) [![Crates.io](https://img.shields.io/crates/v/minimax.svg)](https://crates.io/crates/minimax)

## About

This library provides interfaces that describe:

1. the rules for two-player, perfect-knowledge games;
2. methods of evaluating particular game states for a player; and
3. strategies for choosing moves for a player.

The eventual goal is to have multiple proper strategies, so that any combination
of evaluators and strategies can be tested against each other. Currently, only
a basic alpha-beta pruning Negamax strategy is implemented.

## Example

The `ttt` module contains an implementation of Tic-Tac-Toe, demonstrating how to
use the game and evaluation interfaces. `test` shows how to use strategies.

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
