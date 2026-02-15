"""
Experiment Analysis Script
--------------------------
Computes accuracy metrics (BERTScore, BLEU, ROUGE-L) and generates
latency/accuracy plots from Exp1 (router comparison) and optionally
Exp2 (monolith baseline) CSV results.

Usage:
  python analyze.py --exp1-csv data/exp1_router_comparison/*/results.csv --out-dir /tmp/plots
  python analyze.py --exp1-csv exp1.csv --exp2-csv exp2.csv --out-dir ./plots
"""

import os, sys, argparse, glob

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from evaluator.accuracy_metrics import bertscore_score, bleu_score, rouge_score_fn


def load_csv(pattern: str) -> pd.DataFrame:
    """Load one or more CSVs matching a glob pattern into a single DataFrame."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match: {pattern}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(paths)} file(s): {paths}")
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row BLEU, ROUGE-L, and per-group BERTScore F1 columns."""
    # Drop rows with missing responses or oracle
    mask = df["response"].notna() & df["oracle_response"].notna()
    mask &= ~df["response"].astype(str).str.startswith("ERROR:")
    df = df[mask].copy()
    df["response"] = df["response"].astype(str)
    df["oracle_response"] = df["oracle_response"].astype(str)

    # BLEU (per row)
    df["bleu"] = df.apply(
        lambda r: bleu_score(r["response"], r["oracle_response"]), axis=1
    )

    # ROUGE-L F1 (per row)
    df["rouge_l_f1"] = df.apply(
        lambda r: rouge_score_fn(r["response"], r["oracle_response"])["rougeL"].fmeasure,
        axis=1,
    )

    # BERTScore F1 (batch per router for efficiency)
    df["bertscore_f1"] = 0.0
    for router_name, group in df.groupby("router"):
        candidates = group["response"].tolist()
        references = group["oracle_response"].tolist()
        _, _, f1_list = bertscore_score(candidates, references)
        f1_vals = [float(v) for v in f1_list]
        df.loc[group.index, "bertscore_f1"] = f1_vals

    return df


def plot_cdf(df: pd.DataFrame, col: str, xlabel: str, title: str,
             out_path: str):
    """CDF plot per router strategy with p50/p95 markers."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (router, group) in enumerate(df.groupby("router")):
        color = colors[i % len(colors)]
        sorted_vals = np.sort(group[col].values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, label=router, color=color)

        p50 = np.percentile(sorted_vals, 50)
        p95 = np.percentile(sorted_vals, 95)
        plt.axvline(p50, linestyle="--", color=color, alpha=0.7)
        plt.axvline(p95, linestyle=":", color=color, alpha=0.7)

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="--", label="p50"),
        Line2D([0], [0], color="black", linestyle=":", label="p95"),
    ]
    handles = plt.gca().get_legend_handles_labels()[0] + custom_lines
    plt.legend(title="Router", handles=handles)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative Probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_accuracy_bars(df: pd.DataFrame, out_path: str):
    """Bar chart of mean BERTScore F1 per router."""
    means = df.groupby("router")["bertscore_f1"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(means.index, means.values, color=plt.cm.tab10.colors[:len(means)])
    for bar, val in zip(bars, means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Mean BERTScore F1")
    plt.title("Accuracy by Router Strategy (BERTScore F1)")
    plt.ylim(0, max(means.values.max() + 0.05, 1.0))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_accuracy_by_domain(df: pd.DataFrame, out_path: str):
    """Grouped bar chart: router x domain for BERTScore F1."""
    pivot = df.groupby(["query_type", "router"])["bertscore_f1"].mean().unstack()
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_ylabel("Mean BERTScore F1")
    ax.set_xlabel("Query Domain")
    ax.set_title("Accuracy by Domain and Router Strategy")
    ax.legend(title="Router")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_dit_vs_monolith(exp1: pd.DataFrame, exp2: pd.DataFrame, out_path: str):
    """Side-by-side comparison bars: DIT (best router) vs Monolith."""
    # Pick best DIT router by BERTScore
    dit_means = exp1.groupby("router")["bertscore_f1"].mean()
    best_router = dit_means.idxmax()
    dit_best = exp1[exp1["router"] == best_router]

    labels = ["BERTScore F1", "BLEU", "ROUGE-L F1", "Total Latency (ms)"]
    dit_vals = [
        dit_best["bertscore_f1"].mean(),
        dit_best["bleu"].mean(),
        dit_best["rouge_l_f1"].mean(),
        dit_best["total_latency_ms"].median(),
    ]
    mono_vals = [
        exp2["bertscore_f1"].mean(),
        exp2["bleu"].mean(),
        exp2["rouge_l_f1"].mean(),
        exp2["total_latency_ms"].median(),
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, dit_vals, width, label=f"DIT ({best_router})")
    bars2 = ax.bar(x + width / 2, mono_vals, width, label="Monolith")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("DIT vs Monolith Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def print_summary(df: pd.DataFrame, label: str = ""):
    """Print a summary table to stdout."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    summary = df.groupby("router").agg(
        count=("id", "count"),
        bertscore_f1_mean=("bertscore_f1", "mean"),
        bleu_mean=("bleu", "mean"),
        rouge_l_f1_mean=("rouge_l_f1", "mean"),
        routing_latency_p50=("routing_latency_ms", "median"),
        routing_latency_p95=("routing_latency_ms", lambda x: np.percentile(x, 95)),
        total_latency_p50=("total_latency_ms", "median"),
        total_latency_p95=("total_latency_ms", lambda x: np.percentile(x, 95)),
    )

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(summary.to_string())


def main():
    ap = argparse.ArgumentParser(description="Analyze DIT experiment results")
    ap.add_argument("--exp1-csv", required=True,
                    help="Glob pattern for Exp1 results CSV(s)")
    ap.add_argument("--exp2-csv", default=None,
                    help="Glob pattern for Exp2 (monolith) results CSV(s)")
    ap.add_argument("--out-dir", default="./analysis_output",
                    help="Directory for output plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("[analyze] Loading Exp1 data ...")
    exp1 = load_csv(args.exp1_csv)

    exp2 = None
    if args.exp2_csv:
        print("[analyze] Loading Exp2 data ...")
        exp2 = load_csv(args.exp2_csv)

    # Compute metrics
    print("[analyze] Computing accuracy metrics (this may take a while) ...")
    exp1 = compute_metrics(exp1)
    if exp2 is not None:
        exp2 = compute_metrics(exp2)

    # Plots
    print("[analyze] Generating plots ...")

    plot_cdf(exp1, "total_latency_ms", "Total Latency (ms)",
             "CDF of Total Latency per Router Strategy",
             os.path.join(args.out_dir, "cdf_total_latency.png"))

    plot_cdf(exp1, "routing_latency_ms", "Routing Latency (ms)",
             "CDF of Routing Latency per Router Strategy",
             os.path.join(args.out_dir, "cdf_routing_latency.png"))

    plot_accuracy_bars(exp1, os.path.join(args.out_dir, "accuracy_bertscore.png"))

    plot_accuracy_by_domain(exp1, os.path.join(args.out_dir, "accuracy_by_domain.png"))

    if exp2 is not None:
        plot_dit_vs_monolith(exp1, exp2,
                             os.path.join(args.out_dir, "dit_vs_monolith.png"))

    # Summary
    print_summary(exp1, "Exp1: Router Comparison")
    if exp2 is not None:
        print_summary(exp2, "Exp2: Monolith Baseline")

    print(f"\n[analyze] Done. Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
