"""
Experiment Analysis Script
--------------------------
Uses LLM-as-judge (OpenAI API) to score response accuracy, and generates
latency/accuracy plots from Exp1 (router comparison) and optionally
Exp2 (monolith baseline) CSV results.

Usage:
  export OPENAI_API_KEY=sk-...
  python analyze.py --exp1-csv data/exp1_router_comparison/*/results.csv --out-dir ./plots
  python analyze.py --exp1-csv exp1.csv --exp2-csv exp2.csv --out-dir ./plots

Skip scoring (just plot latency from already-scored CSV):
  python analyze.py --exp1-csv scored_exp1.csv --skip-scoring --out-dir ./plots
"""

import os, sys, argparse, glob, json, time

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


JUDGE_PROMPT = """\
You are an impartial judge evaluating response quality.

Given a question, a candidate response, and a reference (oracle) response,
score the candidate on a scale of 1-5:
  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Somewhat correct but missing key information
  4 = Mostly correct with minor issues
  5 = Fully correct and complete

Question: {question}

Reference answer: {oracle}

Candidate response: {response}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


def load_csv(pattern: str) -> pd.DataFrame:
    """Load one or more CSVs matching a glob pattern into a single DataFrame."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match: {pattern}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {len(paths)} file(s): {paths}")
    return df


def score_with_llm(df: pd.DataFrame, client, model: str,
                   batch_size: int = 20) -> pd.DataFrame:
    """Add 'llm_score' (1-5) and 'llm_reason' columns via LLM-as-judge."""
    mask = df["response"].notna() & df["oracle_response"].notna()
    mask &= ~df["response"].astype(str).str.startswith("ERROR:")
    df = df[mask].copy()
    df["response"] = df["response"].astype(str)
    df["oracle_response"] = df["oracle_response"].astype(str)

    scores = []
    reasons = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        prompt = JUDGE_PROMPT.format(
            question=row["query"],
            oracle=row["oracle_response"],
            response=row["response"],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            parsed = json.loads(text)
            scores.append(int(parsed["score"]))
            reasons.append(parsed.get("reason", ""))
        except Exception as e:
            scores.append(0)
            reasons.append(f"scoring error: {e}")

        if (i + 1) % batch_size == 0:
            print(f"    Scored {i + 1}/{total} ...")

    df["llm_score"] = scores
    df["llm_reason"] = reasons
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
    """Bar chart of mean LLM score per router."""
    means = df.groupby("router")["llm_score"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(means.index, means.values, color=plt.cm.tab10.colors[:len(means)])
    for bar, val in zip(bars, means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Mean LLM Judge Score (1-5)")
    plt.title("Response Accuracy by Router Strategy")
    plt.ylim(0, 5.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_accuracy_by_domain(df: pd.DataFrame, out_path: str):
    """Grouped bar chart: router x domain for LLM score."""
    pivot = df.groupby(["query_type", "router"])["llm_score"].mean().unstack()
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_ylabel("Mean LLM Judge Score (1-5)")
    ax.set_xlabel("Query Domain")
    ax.set_title("Response Accuracy by Domain and Router Strategy")
    ax.legend(title="Router")
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_dit_vs_monolith(exp1: pd.DataFrame, exp2: pd.DataFrame, out_path: str):
    """Side-by-side comparison bars: DIT (best router) vs Monolith."""
    dit_means = exp1.groupby("router")["llm_score"].mean()
    best_router = dit_means.idxmax()
    dit_best = exp1[exp1["router"] == best_router]

    labels = ["LLM Score (1-5)", "Total Latency p50 (ms)"]
    dit_vals = [
        dit_best["llm_score"].mean(),
        dit_best["total_latency_ms"].median(),
    ]
    mono_vals = [
        exp2["llm_score"].mean(),
        exp2["total_latency_ms"].median(),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax = axes[0]
    x = np.arange(1)
    width = 0.35
    b1 = ax.bar(x - width/2, [dit_vals[0]], width, label=f"DIT ({best_router})")
    b2 = ax.bar(x + width/2, [mono_vals[0]], width, label="Monolith")
    ax.set_ylabel("Mean LLM Judge Score")
    ax.set_title("Accuracy: DIT vs Monolith")
    ax.set_xticks(x)
    ax.set_xticklabels(["LLM Score"])
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b in [b1, b2]:
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    # Latency comparison
    ax = axes[1]
    b1 = ax.bar(x - width/2, [dit_vals[1]], width, label=f"DIT ({best_router})")
    b2 = ax.bar(x + width/2, [mono_vals[1]], width, label="Monolith")
    ax.set_ylabel("Median Total Latency (ms)")
    ax.set_title("Latency: DIT vs Monolith")
    ax.set_xticks(x)
    ax.set_xticklabels(["p50 Latency"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b in [b1, b2]:
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

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
        llm_score_mean=("llm_score", "mean"),
        llm_score_std=("llm_score", "std"),
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
    ap.add_argument("--judge-model", default="gpt-4o-mini",
                    help="OpenAI model for LLM-as-judge scoring")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="Skip LLM scoring (use existing llm_score column)")
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("[analyze] Loading Exp1 data ...")
    exp1 = load_csv(args.exp1_csv)

    exp2 = None
    if args.exp2_csv:
        print("[analyze] Loading Exp2 data ...")
        exp2 = load_csv(args.exp2_csv)

    # Score with LLM
    if not args.skip_scoring:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY env var or pass --api-key")
            sys.exit(1)

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print(f"[analyze] Scoring Exp1 with {args.judge_model} ...")
        exp1 = score_with_llm(exp1, client, args.judge_model)

        # Save scored CSV so you don't have to re-score
        scored_path = os.path.join(args.out_dir, "exp1_scored.csv")
        exp1.to_csv(scored_path, index=False)
        print(f"  Saved scored CSV: {scored_path}")

        if exp2 is not None:
            print(f"[analyze] Scoring Exp2 with {args.judge_model} ...")
            exp2 = score_with_llm(exp2, client, args.judge_model)
            scored_path = os.path.join(args.out_dir, "exp2_scored.csv")
            exp2.to_csv(scored_path, index=False)
            print(f"  Saved scored CSV: {scored_path}")

    # Plots
    print("[analyze] Generating plots ...")

    plot_cdf(exp1, "total_latency_ms", "Total Latency (ms)",
             "CDF of Total Latency per Router Strategy",
             os.path.join(args.out_dir, "cdf_total_latency.png"))

    plot_cdf(exp1, "routing_latency_ms", "Routing Latency (ms)",
             "CDF of Routing Latency per Router Strategy",
             os.path.join(args.out_dir, "cdf_routing_latency.png"))

    plot_accuracy_bars(exp1, os.path.join(args.out_dir, "accuracy_scores.png"))

    plot_accuracy_by_domain(exp1, os.path.join(args.out_dir, "accuracy_by_domain.png"))

    if exp2 is not None:
        # Combined CDF with monolith included
        combined = pd.concat([exp1, exp2], ignore_index=True)
        plot_cdf(combined, "total_latency_ms", "Total Latency (ms)",
                 "CDF of Total Latency: All Strategies + Monolith",
                 os.path.join(args.out_dir, "cdf_total_latency_combined.png"))

        plot_dit_vs_monolith(exp1, exp2,
                             os.path.join(args.out_dir, "dit_vs_monolith.png"))

    # Summary
    print_summary(exp1, "Exp1: Router Comparison")
    if exp2 is not None:
        print_summary(exp2, "Exp2: Monolith Baseline")

    print(f"\n[analyze] Done. Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
