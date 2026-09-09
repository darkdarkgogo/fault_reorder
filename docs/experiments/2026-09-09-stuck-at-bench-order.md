# Stuck-at fault-order experiment on BENCH input

## Setup

- PODEM reads the `.bench` files in `PODEM/sample_circuits` directly.
- Only stuck-at ATPG is executed.
- Backtrack limit is 3000, both as the source default and explicitly via
  `-bt 3000` in the experiment.
- Native and SCOAP fault orders are compared with all other settings unchanged.
- No dynamic or static compression option is enabled.

Commands:

```powershell
PODEM/src/atpg.exe -bt 3000 PODEM/sample_circuits/c432_binary.bench
PODEM/src/atpg.exe -bt 3000 -scoap PODEM/sample_circuits/c432_binary.bench
PODEM/src/atpg.exe -bt 3000 PODEM/sample_circuits/c432.bench
PODEM/src/atpg.exe -bt 3000 -scoap PODEM/sample_circuits/c432.bench
```

## Results

| BENCH | Order | Gates | Collapsed faults | Uncollapsed faults | Detected | Coverage | Patterns | Aborted | Redundant | PODEM calls | Backtracks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c432_binary.bench` | native | 288 | 616 | 1192 | 1145 | 96.06% | 92 | 11 | 30 | 133 | 42969 |
| `c432_binary.bench` | SCOAP | 288 | 616 | 1192 | 1145 | 96.06% | 94 | 11 | 30 | 135 | 42985 |
| `c432.bench` | native | 232 | 560 | 1080 | 1034 | 95.74% | 89 | 10 | 30 | 129 | 39967 |
| `c432.bench` | SCOAP | 232 | 560 | 1080 | 1034 | 95.74% | 96 | 10 | 30 | 136 | 39985 |

For both sample BENCH files, changing from the native order to SCOAP leaves the
detected count and coverage unchanged. It changes pattern count and solver
work. The native order is better on pattern count in these two comparisons.

This is evidence for these two orderings, not a proof for every permutation.
The solver currently reports CPU time with only 0.1-second precision, so the
table uses deterministic call and backtrack counts instead.

This earlier comparison regenerated faults directly on each physical BENCH.
The selected logical model now follows the companion V3 map generated from the
original c432 gate boundaries: PODEM executes the binary BENCH with 533 mapped
collapsed faults, including 72 logical XOR-input faults, and an uncollapsed
total of 864. Private XOR expansion nodes and `__smartatpg_bin_*` helpers do not
become fault actions in that mapped run.
