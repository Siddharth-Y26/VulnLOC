# configs/gnn/

The structural branch (Phase 4). `default.yaml` carries every knob; `gcn.yaml`,
`gat.yaml` and `ggnn.yaml` inherit it and change only `kind`, so a sweep across the three
changes the convolution and nothing else. Each resolves to a `StructuralEncoder` in
`src/vulnloc/models/structural/`.

## This group is NOT in `config.yaml`'s `defaults` list, and that is deliberate

Select it explicitly: `+gnn=gcn`, not `gnn=gcn`.

Adding a group to `defaults` changes the resolved config of **every** run, including
semantic ones — which changes their `config_hash`, which is what
`scripts/report_seeds.py` uses to decide whether two runs are replicates of one
experiment. Phase 3's seed sweep is unfinished (seed 42 is logged, seeds 1 and 2 are
not), so adding `gnn` to the defaults right now would make the finished seed
incomparable with the ones still to run, and the seed report would correctly refuse to
aggregate them.

**Move it into `defaults` once Phase 3's three seeds are logged.** It costs one line and
removes the `+`.
