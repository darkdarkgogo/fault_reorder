# Random-order compression erase-skip fix

## Goal

Ensure `random_order_fault_sim()` evaluates every vector exactly once in its shuffled order, including the vector that moves into the current position after an erase.

## Design

Replace the incrementing `for` loop with a manually advanced index:

- If the current vector is redundant, erase it and keep the same index so the shifted vector is evaluated next.
- If the current vector is retained, increment the index.
- Use `size_t` for the index so its type matches `vectors.size()`.

This is preferable to decrementing `i` after erase because it avoids a special case at index zero, and preferable to an iterator loop because the existing code and fault-simulation call are index-oriented.

## Scope

Only the traversal in `ATPG::random_order_fault_sim()` changes. Shuffling, seed updates, fault-list resets, reverse-order compression, TDF behavior, and RL behavior remain unchanged.

## Verification

- Add a regression test that demonstrates consecutive redundant vectors are both removed rather than skipping the shifted vector.
- Rebuild the native extension.
- Run the existing regression suite.
