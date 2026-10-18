# EQUIP

Implementation of closed-loop causal influence governance with role-conditioned influence corridors.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Preparation generates frozen categorical panels with disjoint competence-training and test decisions. To import recorded panels, add `--input panels.json`; records contain `id`, `group`, `split` (`train`/`test`), `messages` (agent × option), `label`, `manipulator` and `attack_target`.

```bash
equip prepare --config configs/reproduce.json --data data/panels
```

## Training and execution

`configs/reproduce.json` fixes seed `17`. Keep this config for every step; change `seed` there and use separate data/output directories for independent runs.

```bash
equip train --config configs/reproduce.json --data data/panels --output checkpoints
equip run --config configs/reproduce.json --data data/panels --checkpoint checkpoints/quality.json --output runs
```

## Evaluation and comparisons

`compare` executes the available methods, including component controls, on the same prepared inputs.

```bash
equip compare --config configs/reproduce.json --data data/panels --checkpoint checkpoints/quality.json --output runs
equip plot --output runs
```

## Outputs

`runs/<method>/` contains `rounds.csv`, `trace.json` and `metrics.json`. Comparisons write `runs/comparison.csv` and `runs/comparison.json`; plots go to `runs/figures/`. `checkpoints/quality.json` stores fitted competence and training groups.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
