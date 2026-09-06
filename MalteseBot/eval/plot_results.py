"""plot_results.py — Generate Chapter 4 figures from the evaluation output.

Reads eval/summary.json and writes PNG charts:
    fig1_accuracy_by_condition.png    keyword precision  RAG vs Baseline
    fig2_hallucination_by_condition.png  hallucination rate RAG vs Baseline
    fig3_accuracy_by_language.png     keyword precision by language x condition
    fig4_accuracy_by_difficulty.png   keyword precision by difficulty x condition

Run (inside the venv, after evaluate.py):
    .venv\\Scripts\\python.exe eval\\plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).resolve().parent
summary = json.loads((EVAL_DIR / "summary.json").read_text(encoding="utf-8"))

RAG_COLOR = "#1a7a3e"
BASE_COLOR = "#a02020"


def _save(fig, name):
    path = EVAL_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path.name)


# Fig 1 — accuracy by condition
def fig_accuracy():
    o = summary["overall"]
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [o["RAG"]["keyword_precision"], o["Baseline"]["keyword_precision"]]
    ax.bar(["RAG", "Baseline"], vals, color=[RAG_COLOR, BASE_COLOR])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Keyword / fact precision")
    ax.set_title("Answer accuracy: RAG vs Baseline")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    _save(fig, "fig1_accuracy_by_condition.png")


# Fig 2 — hallucination by condition
def fig_hallucination():
    o = summary["overall"]
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [o["RAG"]["hallucination_rate"], o["Baseline"]["hallucination_rate"]]
    ax.bar(["RAG", "Baseline"], vals, color=[RAG_COLOR, BASE_COLOR])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Hallucination rate: RAG vs Baseline")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    _save(fig, "fig2_hallucination_by_condition.png")


def _grouped(section, title, fname, keys):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(keys))
    w = 0.35
    rag = [summary[section][k]["RAG"]["keyword_precision"] for k in keys]
    base = [summary[section][k]["Baseline"]["keyword_precision"] for k in keys]
    ax.bar([i - w / 2 for i in x], rag, w, label="RAG", color=RAG_COLOR)
    ax.bar([i + w / 2 for i in x], base, w, label="Baseline", color=BASE_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.upper() if len(k) == 2 else k.capitalize() for k in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Keyword / fact precision")
    ax.set_title(title)
    ax.legend()
    _save(fig, fname)


def main():
    fig_accuracy()
    fig_hallucination()
    _grouped("by_language", "Accuracy by language", "fig3_accuracy_by_language.png", ["en", "mt"])
    _grouped("by_difficulty", "Accuracy by difficulty", "fig4_accuracy_by_difficulty.png", ["easy", "medium", "hard"])


if __name__ == "__main__":
    main()
