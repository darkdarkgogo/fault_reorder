# Offline fault-order training data

This directory contains the pretrained 257-dimensional fault embeddings and
the compact metadata required by `configs/all_benchmarks.json`. They are
committed so an isolated Linux training host does not need DeepGate2, network
access, or an embedding-generation step.

The arrays were generated with the official `python-deepgate` revision and
checkpoint hashes recorded in each NPZ/JSON file. Packaging replaces local
Windows source paths with stable ASCII labels so provenance validation is
portable. To recreate the compact bundle after an authorized export, run:

```bash
python scripts/compact_fault_order_metadata.py \
  --manifest configs/all_benchmarks.json
```
