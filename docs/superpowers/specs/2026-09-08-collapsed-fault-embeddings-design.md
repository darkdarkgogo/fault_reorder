# Collapsed fault embedding export

This implements the revised design approved in the conversation: two-input
gates and swap-invariant functional features. GI uses
`[receiver_hf, connected_signal_hf, sa_value]`; GO uses
`[driver_hf, driver_hf, sa_value]`.
It supersedes the larger draft's fault feature layout for this implementation.
No RL, scoring MLP, training, fault ordering, or ATPG solver changes are included.

## Inputs and identity

Use a two-input BENCH netlist (unary NOT/BUF also supported) and its existing
SMARTATPG_FAULT_MAP_V2 companion. Preserve the ordered collapsed fault IDs and
equivalence counts; do not regenerate faults on the transformed graph.
The original IDs may mention GI2 or higher: these describe the pre-conversion
gate. Resolve the actual binary receiver and source wire from the mapping.
Compute the binary input index in PODEM's level-sorted input order; retain both
BENCH pin order and ATPG pin order in the exported metadata. Some bundled maps
preserve `@dup` records for tied equal inputs at the same mapped location. Keep
both IDs and rows; they intentionally receive the same fault embedding.

GO duplicates the functional embedding of its driver gate's output anchor. GI
uses the functional embedding of its mapped receiver gate's output anchor and
the functional embedding of the signal connected to the faulted input. PI dummy
gates map to the matching INPUT declaration; PO dummy gates have explicit
output-boundary anchors.
Only faults explicitly present in the companion map get rows. A mapped binary
helper may provide a receiver embedding, but creates no new candidate fault.

## Representation and inference

Build a deterministic AND/NOT graph without logic optimization, retaining one
output anchor per input BENCH signal. This is a feature graph only. Reject
multi-input gates, malformed maps, undefined wires, cycles, duplicate IDs, and
unresolvable fault sites.

Use the official python-deepgate inference implementation and its pretrained
checkpoint. Use only hf (128 dimensions); do not include hs or a pin-position
one-hot. Concatenate two hf vectors and SA0=0 / SA1=1, giving 257 dimensions.
No random initialization fallback or untrained projection is allowed. Require
matching encoder weights; record checkpoint SHA256, model source revision,
graph digest, and seed.

## Deliverables

- `python -m fault_embedding prepare`: netlist + fault map -> graph and mapped
  collapsed fault metadata; works without PyTorch.
- `python -m fault_embedding export`: run pretrained inference and export
  numeric NPZ arrays plus JSON with stable fault IDs, row numbers, anchors,
  pin indices, polarity, counts, provenance and feature layout.
- Readable usage instructions and behavioral tests for fanout branches,
  swapped inputs, all supported gate truth tables, real c432 data, and invalid
  input/provenance rejection. Pretrained numerical inference is tested only
  when the user-approved dependency environment is available.

## Implementation sequence

1. Parse and validate binary BENCH; construct and verify the AND/NOT graph.
2. Resolve existing fault-map identities to receiver/driver anchors and pins.
3. Compose paired functional/polarity features and implement CLI/NPZ output.
4. Connect strict official pretrained inference and document installation.
5. Test mappings, request independent code review, fix findings, and run the
   available end-to-end paths. Clearly report any missing runtime dependency.
