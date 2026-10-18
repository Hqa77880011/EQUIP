# EQUIP

Implementation of closed-loop causal influence governance with role-conditioned influence corridors.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Frozen panel inputs

`equip prepare --input panels.json` imports records containing `id`, independent `group`, `split` (`train` or `test`), `messages` [agent,option evidence logits], `label`, `manipulator` and `attack_target`. Frozen option evidence can be exported from agent transcripts on [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) and [GPQA](https://github.com/idavidrein/gpqa). Agent/model inference is upstream; the aggregation and randomized interventions are deterministic given recorded messages and random seed. Without input, preparation creates a generated heterogeneous-quality panel workload with disjoint train/test decisions.

```bash
equip prepare --data data/panels
equip train --data data/panels --output checkpoints
equip run --data data/panels --checkpoint checkpoints/quality.json --output runs
equip compare --data data/panels --checkpoint checkpoints/quality.json --output runs
equip plot --output runs
```

## Measure, regulate, re-measure

`train` estimates externally validated role competence from training decisions and records their groups. It does not infer influence entitlements. Lower/upper corridors and style biases are public config inputs. On a sampled probe, one message is independently included or replaced by the zero-logit substitution with known propensity p. The signed estimator `(z-p)*q/(p*(1-p))` is divided by probe sampling probability, giving an unbiased sparse measurement. The declared q is the aggregate probability assigned to that round's original collective choice. It uses no evaluation label.

Upper and lower virtual queues accumulate their respective residuals. Projected finite-difference ascent on the weight simplex maximizes training-competence utility minus debt-weighted predicted influence. Candidate influence predictions use exact leave-one-message-out aggregation of the frozen evidence; these inexpensive arithmetic predictions are distinct from the sampled intervention observations that update the queues. Corridor violation is evaluated from exact counterfactual aggregation and is never fed to the queue update. No online regret theorem is assumed for this finite action controller.

Comparisons include majority votes, style-weighted static debate aggregation, validated static competence weights, randomized audit without feedback, equal corridors and removal of lower debt. The debate and audit controls operate on the supplied frozen panel evidence; they do not generate additional debate transcripts. Manipulation success is conditional on an incorrect attack target. Probe cost is intervention calls divided by agent-rounds; raw call counts are also saved.

## Configuration and outputs

`configs/default.json` sets horizon, seed, role corridors, probe rate, inclusion propensity, debt and primal steps, and generated panel quality/style. The main `rounds.csv` records utility, attacks, probe calls/cost and queue totals; `trace.json` retains weights, signed influence estimates, evaluation counterfactual effects and debts. `metrics.json` summarizes collective utility, manipulation success, probe cost and cumulative corridor violation. The trace stores decision-time `weights` alongside `next_weights` and the actual controller bounds. The equal-corridor control equalizes both lower and upper bounds. All methods are scored against the original declared role corridors, so ablation violations remain comparable. Comparison CSV/JSON and PNG/SVG plots are computed from run outputs. Imported panel counts must match the role arrays. Training/test group overlap is rejected.

## Registered compact reproduction setting

Six-agent frozen categorical panels, disjoint generated competence-training and decision groups, fixed role corridors and 200 decision rounds.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
equip prepare --config configs/reproduce.json --data data/reproduce
equip train --config configs/reproduce.json --data data/reproduce --output checkpoints/reproduce
equip compare --config configs/reproduce.json --data data/reproduce --checkpoint checkpoints/reproduce/quality.json --output runs/reproduce
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
equip-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
