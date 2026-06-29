"""
experiments/run.py
------------------
Esegue i quattro metodi e produce grafici e tabelle.

USO:
    python experiments/run.py [--skip-a] [--budget-fraction 0.05]
"""

import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.data import read_photo_matrix, read_query_log, cosine_matrix, compute_utility
from algorithms.brute_force import run as run_a
from algorithms.indepdf     import run as run_b
from algorithms.shapley     import run as run_c
from algorithms.ewc         import run as run_d

OUT = ROOT / "experiments" / "output"
OUT.mkdir(exist_ok=True)

COLORS = {"A": "#2166ac", "B": "#f4a582", "C": "#1a9850", "D": "#d6604d"}


def evaluate(kept, queries, sim, max_q=None):
    return compute_utility(set(kept), queries, sim, max_q)


def run_all(budget_fraction=0.5, k_shapley=3, skip_a=False, max_q=None):
    photos  = read_photo_matrix()
    queries = read_query_log()
    N = photos.shape[0]
    budget = max(1, int(N * budget_fraction))

    print(f"\nN={N} foto | {len(queries)} query | budget={budget}\n")

    print("Calcolo matrice similarity ...")
    t0 = time.time()
    sim = cosine_matrix(photos)
    print(f"  fatto in {time.time()-t0:.1f}s\n")

    results = {}

    # ── A ─────────────────────────────────────────────────────────
    if skip_a:
        print("[A] Saltato (intrattabile per N grande)")
        results["A"] = {"kept": list(range(N-3)), "utility": None,
                        "time": None, "n_kept": N-3}
    else:
        t0 = time.time()
        ra = run_a(photos, queries, budget=N-3, max_q=max_q, sim=sim)
        ta = time.time() - t0
        ua = evaluate(ra["kept"], queries, sim, max_q)
        results["A"] = {**ra, "utility": ua, "time": ta,
                        "n_kept": len(ra["kept"])}
        print(f"[A] done in {ta:.3f}s\n")

    # ── B ─────────────────────────────────────────────────────────
    t0 = time.time()
    rb = run_b(photos, queries, budget=budget)
    tb = time.time() - t0
    ub = evaluate(rb["kept"], queries, sim, max_q)
    results["B"] = {**rb, "utility": ub, "time": tb,
                    "n_kept": len(rb["kept"])}
    print(f"[B] done in {tb:.3f}s\n")

    # ── C ─────────────────────────────────────────────────────────
    t0 = time.time()
    rc = run_c(photos, queries, budget=k_shapley, max_q=max_q, sim=sim)
    tc = time.time() - t0
    uc = evaluate(rc["kept"], queries, sim, max_q)
    results["C"] = {**rc, "utility": uc, "time": tc,
                    "n_kept": len(rc["kept"])}
    print(f"[C] done in {tc:.3f}s\n")

    # ── D ─────────────────────────────────────────────────────────
    t0 = time.time()
    rd = run_d(photos, queries, budget=budget, sim=sim)
    td = time.time() - t0
    ud = evaluate(rd["kept"], queries, sim, max_q)
    results["D"] = {**rd, "utility": ud, "time": td,
                    "n_kept": len(rd["kept"])}
    print(f"[D] done in {td:.3f}s\n")

    _print_summary(results)
    _save_plots(results, N)
    return results


def _print_summary(results):
    print("=" * 55)
    print(f"{'Metodo':<8} {'# tenute':<10} {'Utility':<14} {'Tempo (s)'}")
    print("-" * 55)
    for m in ["A", "B", "C", "D"]:
        r = results[m]
        u = f"{r['utility']:.6f}" if r["utility"] is not None else "N/A"
        t = f"{r['time']:.3f}"   if r["time"]    is not None else "N/A"
        print(f"  {m:<6} {r['n_kept']:<10} {u:<14} {t}")
    print("=" * 55)

    with open(OUT / "summary.txt", "w") as f:
        f.write(f"{'Metodo':<8} {'# tenute':<10} {'Utility':<14} {'Tempo (s)'}\n")
        f.write("-" * 50 + "\n")
        for m in ["A", "B", "C", "D"]:
            r = results[m]
            u = f"{r['utility']:.6f}" if r["utility"] is not None else "N/A"
            t = f"{r['time']:.3f}"   if r["time"]    is not None else "N/A"
            f.write(f"  {m:<6} {r['n_kept']:<10} {u:<14} {t}\n")


def _save_plots(results, N):
    methods_with_u = [m for m in "ABCD" if results[m]["utility"] is not None]
    utils  = [results[m]["utility"] for m in methods_with_u]
    times  = [results[m]["time"]    for m in methods_with_u]
    n_kept = [results[m]["n_kept"]  for m in methods_with_u]

    # Utility bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods_with_u, utils,
                  color=[COLORS[m] for m in methods_with_u],
                  width=0.5, edgecolor="black")
    for bar, v in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(utils) * 0.01,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Utility f(D')")
    ax.set_xlabel("Metodo")
    ax.set_title("Confronto utility")
    ax.set_ylim(0, max(utils) * 1.15)
    plt.tight_layout()
    plt.savefig(OUT / "utility.png", dpi=150)
    plt.close()

    # Runtime bar chart
    methods_with_t = [m for m in "ABCD" if results[m]["time"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods_with_t,
           [results[m]["time"] for m in methods_with_t],
           color=[COLORS[m] for m in methods_with_t],
           width=0.5, edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Secondi (scala log)")
    ax.set_xlabel("Metodo")
    ax.set_title("Tempo di esecuzione")
    plt.tight_layout()
    plt.savefig(OUT / "runtime.png", dpi=150)
    plt.close()

    # Trade-off scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods_with_u:
        ax.scatter(results[m]["n_kept"], results[m]["utility"],
                   color=COLORS[m], s=120, zorder=3)
        ax.annotate(f"  {m}", (results[m]["n_kept"], results[m]["utility"]),
                    fontsize=10)
    ax.set_xlabel("Foto mantenute")
    ax.set_ylabel("Utility f(D')")
    ax.set_title("Trade-off: spazio vs qualità")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT / "tradeoff.png", dpi=150)
    plt.close()

    # Score distribution B
    if "scores" in results["B"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(results["B"]["scores"], bins=50,
                color=COLORS["B"], edgecolor="none")
        ax.set_title("Distribuzione score IndepDF (Metodo B)")
        ax.set_xlabel("Score")
        ax.set_ylabel("Conteggio")
        plt.tight_layout()
        plt.savefig(OUT / "scores_B.png", dpi=150)
        plt.close()

    print(f"\nGrafici salvati in {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-a",          action="store_true")
    parser.add_argument("--budget-fraction", type=float, default=0.5)
    parser.add_argument("--k-shapley",       type=int,   default=3)
    parser.add_argument("--max-queries",     type=int,   default=None)
    args = parser.parse_args()

    run_all(
        budget_fraction=args.budget_fraction,
        k_shapley=args.k_shapley,
        skip_a=args.skip_a,
        max_q=args.max_queries,
    )