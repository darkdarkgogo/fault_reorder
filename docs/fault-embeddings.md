# Collapsed fault embeddings

The exporter represents every preserved collapsed fault using only DeepGate2's
functional embedding `hf`:

```text
GI: [receiver_gate_hf(128), connected_signal_hf(128), sa_value(1)]
GO: [driver_gate_hf(128), driver_gate_hf(128), sa_value(1)]
```

The output has 257 columns. `sa_value` is 0 for SA0 and 1 for SA1. There is no
structural embedding and no input-position one-hot in the numeric feature. A GI
fault concatenates its receiving binary gate's function with the function on
the faulted input net. A GO fault duplicates its driving gate's function. Thus,
swapping input 0 and input 1 of a commutative two-input gate does not change the
feature for a fault on the same input net. BENCH and PODEM pin indices remain in
JSON as ATPG metadata, but are not model inputs.

The existing `*_binary.faultmap` is the source of collapsed fault identity and
order. No fault is regenerated and no binary helper creates an extra fault.
Bundled maps occasionally contain `@dup` IDs for tied equal inputs that resolve
to the same location; both rows are retained and intentionally have equal
features.

V3 maps also preserve logical input faults of XOR cells whose NAND/NOT
implementation appears in the binary BENCH. Such a fault remains one logical
fault even though the original XOR input feeds two internal paths. Its feature
uses the XOR output anchor as `receiver_gate_hf` and the original input signal
anchor as `connected_signal_hf`; private `W/Z/X/Y` expansion nodes never create
fault rows. For `c432_binary`, the catalog contains 533 collapsed faults,
including 72 logical XOR-input faults, with an uncollapsed total of 864.

Every bundled faultmap is bound to its exact binary BENCH digest in
`fault_embedding/faultmap_bindings.json`. For a new custom pair, create
`<name>.faultmap.binding.json` containing `bench_sha256`, `faultmap_sha256`,
`source_hash`, and `circuit_hash`; copy the two legacy hash values from the
faultmap header.

## Runtime

This workspace uses the official `python-deepgate` source and checkpoint at:

```text
third_party/python-deepgate
third_party/python-deepgate/deepgate/pretrained/model.pth
```

Downloaded source revision:

```text
173db7529cefc97f7b9b2b3fa97ec1bd5773754d
```

Checkpoint SHA256:

```text
9bc4a0c1f8fc57cc3aa0498dd8737af561ca71c26ca5332288d12a91a308f4d5
```

Reproduce the source and checkpoint from the official repository with:

```powershell
git -c core.autocrlf=false clone https://github.com/zshi0616/python-deepgate.git third_party/python-deepgate
git -C third_party/python-deepgate checkout 173db7529cefc97f7b9b2b3fa97ec1bd5773754d
Get-FileHash third_party/python-deepgate/deepgate/pretrained/model.pth -Algorithm SHA256
```

The exporter hashes every Python file under `deepgate/` and requires source
digest `b88833d4bf6979909e33f92b76fb0101e66a6dcdc12ec3b539d0b4a3fa19312c`.
It also requires the checkpoint hash shown above, so a modified checkout or
weight file is rejected before inference.

The local Conda `d2l` environment contains PyTorch 1.11, PyG 2.1 and
torch-scatter 2.0.9. Invoke it directly on this machine:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_embedding export `
  --bench PODEM/sample_circuits/c432_binary.bench `
  --out-dir artifacts/c432-official
```

Use a new output directory for each run. The command refuses to overwrite an
existing artifact so results cannot silently mix.

To validate only parsing and fault mapping without PyTorch:

```powershell
python -m fault_embedding prepare `
  --bench PODEM/sample_circuits/c432_binary.bench `
  --out-dir artifacts/c432-prepared
```

## Outputs

For `c432_binary`, export creates:

- `c432_binary.graph.json`: inspectable AND/NOT feature graph and anchors.
- `c432_binary.faults.json`: ordered fault metadata, pin mappings, equivalence
  counts, feature layout and inference provenance.
- `c432_binary.gates.npz`: named `hf` matrix for inspecting node functions.
- `c432_binary.fault_embeddings.npz`: final matrix plus aligned fault IDs,
  gate/connected-signal anchors, polarities and equivalence counts.

Read the result with NumPy:

```python
import numpy as np

with np.load(
    "artifacts/c432-v3-embeddings/c432_binary.fault_embeddings.npz",
    allow_pickle=False,
) as data:
    fault_ids = data["fault_ids"]       # shape: [533]
    embeddings = data["embeddings"]    # shape: [533, 257]
    print(fault_ids[0], embeddings[0])
```

The metadata records the feature graph digest, checkpoint hash, model source
hash, seed and runtime versions. Every export runs the pinned official model;
there is no cache or random-weight fallback in the production CLI.
