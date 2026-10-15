import argparse
from pathlib import Path
import numpy as np
from .common import load, save, table, plot
from .core import aggregate, influence, control, simplex


def prepare(a, c):
    if a.input:
        records = load(a.input)
    else:
        rng = np.random.default_rng(c["seed"])
        records = []
        for t in range(c["rounds"] + 100):
            label = int(rng.integers(c["choices"]))
            votes = []
            for quality in c["quality"]:
                choice = (
                    label
                    if rng.random() < quality
                    else int(rng.choice([v for v in range(c["choices"]) if v != label]))
                )
                votes.append((4 * np.eye(c["choices"])[choice]).tolist())
            records.append(
                dict(
                    id=str(t),
                    group=str(t),
                    messages=votes,
                    label=label,
                    manipulator=c["agents"] - 1,
                    attack_target=int(np.argmax(votes[-1])),
                    split="train" if t < 100 else "test",
                    source="generated_frozen_panel",
                )
            )
    for r in records:
        if np.asarray(r["messages"]).ndim != 2:
            raise ValueError("messages must have shape agent,option")
    save(Path(a.data) / "panels.json", records)


def train(a, c):
    records = [r for r in load(Path(a.data) / "panels.json") if r["split"] == "train"]
    if not records:
        raise ValueError("No training panels")
    quality = np.mean(
        [
            [np.argmax(message) == r["label"] for message in r["messages"]]
            for r in records
        ],
        axis=0,
    )
    save(
        Path(a.output) / "quality.json",
        dict(quality=quality, training_groups=sorted({r["group"] for r in records})),
    )


def run(a, c, method):
    records = [r for r in load(Path(a.data) / "panels.json") if r["split"] == "test"][
        : c["rounds"]
    ]
    fitted = load(a.checkpoint)
    if {r["group"] for r in records} & set(fitted["training_groups"]):
        raise ValueError("Training/test panel groups overlap")
    quality = np.array(fitted["quality"])
    n = len(quality)
    if len(c["lower"]) != n or len(c["upper"]) != n:
        raise ValueError("Corridors must match panel size")
    rng = np.random.default_rng(c["seed"])
    weights = simplex(quality) if method == "static" else np.full(n, 1 / n)
    plus = np.zeros(n)
    minus = np.zeros(n)
    cumulative = np.zeros(n)
    rows = []
    trace = []
    p = c["inclusion_probability"]
    sampling = c["probe_probability"]
    if not 0 < p < 1 or not 0 < sampling <= 1:
        raise ValueError(
            "Require positive probe coverage and inclusion propensity strictly between 0 and 1"
        )
    for t, r in enumerate(records):
        messages = np.array(r["messages"])
        style = np.ones(n) if method == "majority" else np.array(c["style"])
        if method == "majority":
            votes = np.eye(messages.shape[1])[messages.argmax(1)].sum(0)
            prediction = int(votes.argmax())
        else:
            prediction = int(aggregate(messages, weights, style).argmax())
        outcome = prediction
        estimate = np.zeros(n)
        probes = 0
        for i in range(n):
            if method in ("majority", "debate", "static") or rng.random() >= sampling:
                continue
            z = int(rng.random() < p)
            mask = np.ones(n)
            mask[i] = z
            q = aggregate(messages, weights, style, mask)[outcome]
            estimate[i] = (z - p) / (p * (1 - p)) * q / sampling
            probes += 1
        true_effect = influence(messages, weights, style, outcome)
        cumulative += true_effect
        if method in ("equip", "equal_corridor", "no_lower"):
            upper = (
                np.array(c["upper"])
                if method != "equal_corridor"
                else np.full(n, np.mean(c["upper"]))
            )
            lower = np.zeros(n) if method == "no_lower" else np.array(c["lower"])
            plus = np.maximum(0, plus + c["debt_step"] * (estimate - upper))
            minus = np.maximum(0, minus + c["debt_step"] * (lower - estimate))
            weights = control(
                messages, weights, style, outcome, quality, plus, minus, c["step"]
            )
        attack = int(r["attack_target"])
        attempted = attack != r["label"]
        rows.append(
            dict(
                round=t + 1,
                utility=prediction == r["label"],
                attack_attempt=attempted,
                manipulation_success=attempted and prediction == attack,
                probe_calls=probes,
                probe_cost=probes / n,
                upper_debt=float(plus.sum()),
                lower_debt=float(minus.sum()),
            )
        )
        trace.append(
            dict(
                id=r["id"],
                prediction=prediction,
                weights=weights.copy(),
                estimate=estimate,
                influence=true_effect,
                upper=plus.copy(),
                lower=minus.copy(),
            )
        )
    summary = dict(
        method=method,
        utility=np.mean([r["utility"] for r in rows]),
        manipulation_success=sum(r["manipulation_success"] for r in rows)
        / max(1, sum(r["attack_attempt"] for r in rows)),
        probe_cost=np.mean([r["probe_cost"] for r in rows]),
        corridor_violation=float(
            np.maximum(cumulative - len(rows) * np.array(c["upper"]), 0).sum()
            + np.maximum(len(rows) * np.array(c["lower"]) - cumulative, 0).sum()
        ),
    )
    output = Path(a.output) / method
    table(output / "rounds.csv", rows)
    save(output / "trace.json", trace)
    save(output / "metrics.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train", "run", "compare", "plot"])
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/panels")
    p.add_argument("--output", default="runs")
    p.add_argument("--checkpoint", default="checkpoints/quality.json")
    p.add_argument(
        "--method",
        choices=[
            "equip",
            "majority",
            "debate",
            "static",
            "audit_only",
            "equal_corridor",
            "no_lower",
        ],
        default="equip",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "run":
        run(a, c, a.method)
    elif a.command == "compare":
        rows = [
            run(a, c, m)
            for m in [
                "majority",
                "debate",
                "static",
                "audit_only",
                "equal_corridor",
                "no_lower",
                "equip",
            ]
        ]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
