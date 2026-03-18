# Data

`sample_parts.csv` is a synthetic demo dataset for the nesting-only pipeline.

Expected columns:

- `uid` or any id-like column
- `pid_raw` optional
- one length column: `Fleng_mm`, `Fleng`, `length`, or `L`
- one width column: `Fwidth_mm`, `Fwidth`, `width`, or `W`

If you use your own file, export it to CSV and run:

```bash
python -m experiments.run_demo --parts_csv data/my_parts.csv
```

`run_demo.py` runs the Receding Horizon MCTS pipeline only.

If the CSV belongs to `data/public_binpacktwo/instances/`, `run_demo.py` will
auto-load `instance_meta.csv` and apply the benchmark stock size and rotation
flag. You can still override the stock size explicitly with `--board_w` and
`--board_h`.
