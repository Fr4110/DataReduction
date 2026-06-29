"""
make_plots.py
-------------
Rigenera i grafici del report con stile publication-quality.

USO (con i file reali in data/):
    python make_plots.py

I PNG vengono salvati in experiments/output/
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.data import read_photo_matrix, read_query_log, cosine_matrix, compute_utility
from algorithms.indepdf import score_photos
from algorithms.ewc import _compute_scores as ewc_scores

OUT = ROOT / "experiments" / "output"
OUT.mkdir(exist_ok=True)

# ── stile globale ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})

CB  = "#2166ac"   # B — blu
CG  = "#1a9850"   # C — verde
CR  = "#d6604d"   # D — rosso
CA  = "#4d4d4d"   # A — grigio


def load_real():
    photos  = read_photo_matrix()
    queries = read_query_log()
    return photos, queries


# ── 1. Score distribution B (reale) ─────────────────────────────
def plot_score_dist(photos, queries):
    scores = score_photos(photos, queries)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=60, color=CB, edgecolor="none", alpha=0.85)

    # evidenzia la soglia tra foto inattive e attive
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle=":")
    ax.annotate("30,683 photos\nnever queried",
                xy=(0, ax.get_ylim()[1]*0.9),
                xytext=(0.0008, ax.get_ylim()[1]*0.85),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("IndepDF score")
    ax.set_ylabel("Number of photos")
    ax.set_title("Score distribution — Method B (IndepDF)")
    plt.tight_layout()
    plt.savefig(OUT / "scores_B.png", bbox_inches="tight")
    plt.close()
    print("  scores_B.png salvato")


# ── 2. Utility comparison bar chart ─────────────────────────────
def plot_utility(results_real):
    methods = ["B", "C", "D"]
    utils   = [results_real[m] for m in methods]
    colors  = [CB, CG, CR]
    labels  = ["B — IndepDF", "C — Shapley", "D — EWC"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, utils, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Utility $f(D')$")
    ax.set_ylim(0, max(utils) * 1.2)
    ax.set_title(f"Utility comparison — full dataset ($B=2{{,}}081$)")
    ax.grid(axis="x", alpha=0)
    plt.tight_layout()
    plt.savefig(OUT / "utility.png", bbox_inches="tight")
    plt.close()
    print("  utility.png salvato")


# ── 3. Trade-off scatter ─────────────────────────────────────────
def plot_tradeoff(kept_counts, utils_real):
    fig, ax = plt.subplots(figsize=(7, 4))

    items = [
        ("B", kept_counts["B"], utils_real["B"], CB, "IndepDF"),
        ("C", kept_counts["C"], utils_real["C"], CG, "Shapley ($k=3$)"),
        ("D", kept_counts["D"], utils_real["D"], CR, "EWC"),
    ]
    for m, k, u, c, lbl in items:
        ax.scatter(k, u, color=c, s=160, zorder=4, label=f"{m} — {lbl}")
        ax.annotate(f"  {m}", (k, u), fontsize=10, va="center")

    ax.set_xlabel("Photos retained")
    ax.set_ylabel("Utility $f(D')$")
    ax.set_title("Quality vs.\ storage trade-off")
    ax.legend(loc="center left", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT / "tradeoff.png", bbox_inches="tight")
    plt.close()
    print("  tradeoff.png salvato")


# ── 4. Runtime bar chart ─────────────────────────────────────────
def plot_runtime(times_real):
    methods = ["B", "C", "D"]
    times   = [times_real[m] for m in methods]
    colors  = [CB, CG, CR]
    labels  = ["B — IndepDF", "C — Shapley", "D — EWC"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.15,
                f"{val*1000:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title("Runtime comparison — full dataset")
    ax.grid(axis="x", alpha=0)
    plt.tight_layout()
    plt.savefig(OUT / "runtime.png", bbox_inches="tight")
    plt.close()
    print("  runtime.png salvato")


# ── 5. Synthetic dataset: utility + Shapley bar (side by side) ──
def plot_synthetic():
    synth_data = {
        "A": {"utility": 0.6848, "time": 0.001, "kept": 5},
        "B": {"utility": 0.5683, "time": 0.0001, "kept": 4},
        "C": {"utility": 0.4310, "time": 0.006,  "kept": 3},
        "D": {"utility": 0.5683, "time": 0.0001, "kept": 4},
    }
    shapley = {1:0.1014, 2:0.1111, 3:0.1384, 4:0.1476,
               5:0.1489, 6:0.1386, 7:0.1114, 8:0.1026}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # — utility bar —
    methods = ["A", "B", "C", "D"]
    utils   = [synth_data[m]["utility"] for m in methods]
    cols    = [CA, CB, CG, CR]
    bars    = ax1.bar(methods, utils, color=cols, width=0.5,
                      edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, utils):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Utility $f(D')$")
    ax1.set_ylim(0, max(utils)*1.2)
    ax1.set_title("Utility — synthetic dataset ($N=8$, $B=4$)")
    ax1.grid(axis="x", alpha=0)

    # — Shapley values bar —
    photos_ids = list(shapley.keys())
    phi_vals   = list(shapley.values())
    bar_colors = [CR if v == max(phi_vals) else "#aaaaaa" for v in phi_vals]
    ax2.bar([str(i) for i in photos_ids], phi_vals,
            color=bar_colors, width=0.6, edgecolor="white")
    ax2.set_xlabel("Photo id")
    ax2.set_ylabel("Shapley value $\\phi_d$")
    ax2.set_title("Shapley values — synthetic dataset ($N=8$)")
    ax2.grid(axis="x", alpha=0)

    plt.tight_layout()
    plt.savefig(OUT / "synthetic_combined.png", bbox_inches="tight")
    plt.close()
    print("  synthetic_combined.png salvato")


# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time
    print("Caricamento dati reali ...")
    photos, queries = load_real()

    print("Calcolo scores ...")
    sc_b = score_photos(photos, queries)
    sc_d = ewc_scores(photos, queries, alpha=0.5)

    budget = int(photos.shape[0] * 0.05)

    rank_b = np.argsort(sc_b)[::-1]
    rank_d = np.argsort(sc_d)[::-1]

    kept_b = sorted(rank_b[:budget].tolist())
    kept_d = sorted(rank_d[:budget].tolist())

    print("Calcolo utility (richiede la matrice di similarity) ...")
    sim = cosine_matrix(photos)

    t0 = _time.time(); u_b = compute_utility(set(kept_b), queries, sim); tb = _time.time()-t0
    t0 = _time.time(); u_d = compute_utility(set(kept_d), queries, sim); td = _time.time()-t0

    # C: closed-form Shapley
    from algorithms.shapley import _closed_form
    t0 = _time.time()
    phi = _closed_form(photos, queries)
    tc = _time.time()-t0
    top3 = np.argsort(phi)[::-1][:3].tolist()
    u_c  = compute_utility(set(top3), queries, sim)

    utils_real = {"B": u_b, "C": u_c, "D": u_d}
    times_real = {"B": tb,  "C": tc,  "D": td}
    kept_counts = {"B": budget, "C": 3, "D": budget}

    print(f"\nUtility — B={u_b:.4f}  C={u_c:.4f}  D={u_d:.4f}")
    print(f"Tempo   — B={tb*1000:.1f}ms  C={tc*1000:.1f}ms  D={td*1000:.1f}ms\n")

    print("Generazione grafici ...")
    plot_score_dist(photos, queries)
    plot_utility(utils_real)
    plot_tradeoff(kept_counts, utils_real)
    plot_runtime(times_real)
    plot_synthetic()

    print("\nFatto. File in experiments/output/:")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
