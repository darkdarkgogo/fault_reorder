# Fault ordering RL implementation

Continue the approved listwise policy-gradient design with the existing scorer,
Plackett–Luce policy and ordered PODEM binding. Use the local d2l Python runtime.

1. Strengthen manifest preflight and verify BENCH, map, graph and pretrained
   embedding provenance before any baseline episode.
2. Implement reward/coverage state, equal-circuit REINFORCE updates, deterministic
   evaluation, atomic checkpoints and exact CPU RNG restoration.
3. Add validate/train/evaluate commands, the 16-circuit manifest and reproducible
   pretrained embedding preparation. Export scores and ranks from best.
4. Test policy math, coverage guards, transactional failure, resume equivalence,
   data mismatch and real ordered PODEM. Run a two-benchmark smoke experiment.
5. Review the implementation, fix findings and document commands and measured
   validation. Full 100-round research training is a separate experiment.

The brainstorming skill's optional implementation-plan dependency
(`writing-plans`) is not installed locally; this plan records the same scoped
implementation and validation sequence without that dependency.
