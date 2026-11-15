# Joint G4, zero cost, seed3 — unique final inference export

Outcome: **PERFORMANCE_FAIL**. This artifact does not replace the frozen healthy scalar baseline.
See [the full report](../../docs/JOINT_HEALTHY_TWOHOUR_20261115.md) for all conditions,
repetitions, CPU scope and limitations. Zero decision cost does not optimize call reduction.

Only the unique completed3600-second/actor endpoint is provided here. No intermediate
checkpoint was selected by performance. Tensor data and index are unchanged from the
cloud policy export; metadata is made portable and excludes private launch routes.
Full online/target/BN/optimizer variables are additionally backed up locally. Replay,
historical checkpoints and packet logs remain in the cloud. No complete resume is provided.

Use the same TF1.14/Sonnet1.34 and validated custom kernel/native environment. With a NEW
explicit experiment authorization, from the repository root:

```sh
python joint_healthy.py evaluate --config models/joint-g4-cost0-seed3-20261115/evaluation.json \
  --authorization /path/to/new-authorization.json --run-id new-joint-frozen-check \
  --checkpoint models/joint-g4-cost0-seed3-20261115/endpoint --scenario 0 --seconds 60
```

Twenty-four static trained conditions, three frozen repeats, one training seed.
Reused references were not simultaneously randomized with the final endpoint.
