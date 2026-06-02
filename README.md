# Tiny Transformer Generalization on Filtered Tic-Tac-Toe

Trains tiny decoder-only transformers on tic-tac-toe move-sequences with all
**horizontal-winning moves removed**, then probes whether they (1) score
horizontal wins they never saw (rotational-symmetry generalization) or (2) only
block horizontal threats (a pattern still present in training).

See the original experiment spec and implementation plan in `docs/`.

> **Status:** Experiments ongoing. See `results/` for current outputs by run.

## Requirements

Python 3.10+

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -v
```

## Run the full experiment

```bash
python run_experiment.py
# writes results/raw.json, results/agg.json, results/capacity.png
```

## Results

Results are organized by run in the `results/` folder:

```
results/
  baseline/          # original run
  more_epochs/       # extended training run
```

![Capacity plot](results/capacity.png)

## How it works

- `ttt/board.py` — board + win/line logic
- `ttt/solver.py` — perfect minimax, optimal move sets
- `ttt/enumerate.py` — all 5,478 reachable positions + capped move-orderings
- `ttt/dataset.py` — tokenization + horizontal-win filter (drops offensive targets, keeps blocks)
- `ttt/model.py` — nanoGPT-style decoder
- `ttt/train.py` — training loop
- `ttt/probes.py` / `ttt/evaluate.py` — win/block probes (H/V/D) + metrics
- `ttt/sweep.py` — capacity sweep, seed aggregation, capacity plot
