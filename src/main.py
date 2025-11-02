# run_both_from_json.py
from cfr_experiment import Experiment2 as CFRExperiment     # <- adjust if needed
from dream_experiment import Experiment as DreamExperiment  # <- adjust if needed

import os, glob, csv, time, json
from datetime import datetime
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# --------------------------
# Repro helpers / I/O utils
# --------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_run_csv(outpath: str, iterations, exploitabilities):
    """Standardize per-run CSV format: Iteration, Exploitability"""
    ensure_dir(os.path.dirname(outpath))
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Exploitability"])
        for it, ex in zip(iterations, exploitabilities):
            w.writerow([int(it), f"{float(ex):.6f}"])
    print(f"[CSV] Saved: {outpath}")


# ------------------------------------------------------------
# Aggregate multiple runs into mean/std over iterations
# ------------------------------------------------------------
def aggregate_runs(file_paths, iteration_col="Iteration", value_col="Exploitability"):
    """
    Reads multiple CSVs, aligns by iteration, and returns a DataFrame with:
      [Iteration, mean, std]
    """
    df_all = None
    for k, fp in enumerate(sorted(file_paths)):
        df = pd.read_csv(fp)
        df = df[[iteration_col, value_col]].copy()
        df[iteration_col] = pd.to_numeric(df[iteration_col], errors='coerce')
        df[value_col]     = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[iteration_col, value_col])
        df = df.groupby(iteration_col, as_index=False).mean() 
        df = df.rename(columns={value_col: f'run{k+1}'})
        df_all = df if df_all is None else pd.merge(df_all, df, on=iteration_col, how='outer')

    if df_all is None or df_all.empty:
        raise ValueError("No data found. Check file patterns/paths.")

    df_all = df_all.sort_values(iteration_col)
    run_cols = [c for c in df_all.columns if c.startswith('run')]
    df_all['mean'] = df_all[run_cols].mean(axis=1, skipna=True)
    df_all['std']  = df_all[run_cols].std(axis=1, ddof=1, skipna=True)
    return df_all[[iteration_col, 'mean', 'std']]


def aggregate_and_plot_from_output(root: str, out_svg_name: str = "exploitability_cfr_vs_dream.svg"):
    """
    Re-aggregates from per-run CSVs saved in `root` and plots mean±std,
    """
    # Locate files inside the output path (root)
    dream_files = sorted(glob.glob(os.path.join(root, "DREAM_seed*_run*.csv")))
    cfr_files   = sorted(glob.glob(os.path.join(root, "CFR_seed*_run*.csv")))

    dream_stats_csv = None
    cfr_stats_csv   = None

    # Aggregate DREAM
    dream_stats = None
    if dream_files:
        dream_stats = aggregate_runs(
            dream_files,
            iteration_col="Iteration",
            value_col="Exploitability"
        )
        dream_stats_csv = os.path.join(root, "DREAM_exploitability_mean_std.csv")
        dream_stats.to_csv(dream_stats_csv, index=False)
        print(f"[CSV] Saved aggregated DREAM stats to {dream_stats_csv}")
    else:
        print("[INFO] No DREAM run CSVs found in output path.")

    # Aggregate CFR
    cfr_stats = None
    if cfr_files:
        cfr_stats = aggregate_runs(
            cfr_files,
            iteration_col="Iteration",
            value_col="Exploitability"
        )
        cfr_stats_csv = os.path.join(root, "CFR_exploitability_mean_std.csv")
        cfr_stats.to_csv(cfr_stats_csv, index=False)
        print(f"[CSV] Saved aggregated CFR   stats to {cfr_stats_csv}")
    else:
        print("[INFO] No CFR run CSVs found in output path.")

    # Plot (copying your plotting pattern)
    if (dream_stats is None) and (cfr_stats is None):
        print("[PLOT] Nothing to plot.")
        return

    plt.figure(figsize=(9, 5))

    if dream_stats is not None:
        x_d = dream_stats["Iteration"].values
        y_d = dream_stats["mean"].values
        s_d = dream_stats["std"].values
        plt.plot(x_d, y_d, label="DREAM (mean)")
        plt.fill_between(x_d, y_d - s_d, y_d + s_d, alpha=0.2)

    if cfr_stats is not None:
        x_c = cfr_stats["Iteration"].values
        y_c = cfr_stats["mean"].values
        s_c = cfr_stats["std"].values
        plt.plot(x_c, y_c, label="CFR (mean)")
        plt.fill_between(x_c, y_c - s_c, y_c + s_c, alpha=0.2)

    plt.xlabel("Iterations")
    plt.ylabel("Exploitability")
    plt.title("Exploitability vs Iterations (mean ± std over runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_svg = os.path.join(root, out_svg_name)
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved: {out_svg}")


# --------------------------
# Main benchmark procedure
# --------------------------
def run_cfr_once(seed: int, iterations: int, eval_every: int):
    """Runs one CFR experiment with the given seed; returns (iters, exploits)."""
    set_all_seeds(seed)
    exp = CFRExperiment()
    iters, exploits, _eval_times = exp.run(iterations=iterations, eval_every=eval_every)
    return iters, exploits


def run_dream_once(seed: int, iterations: int, eval_every: int,
                   trajs_per_iter: int, batch_size: int, num_train_steps: int):
    """Runs one DREAM experiment with the given seed; returns (iters, exploits)."""
    set_all_seeds(seed)
    exp = DreamExperiment()
    results = exp.run(seed=seed,
                      iters=iterations,
                      eval_every=eval_every,
                      trajs_per_iter=trajs_per_iter,
                      batch_size=batch_size,
                      num_train_steps=num_train_steps)
    return results['iterations'], results['exploitability']


def load_config(path="benchmark_config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file '{path}' not found. Create it (see example in the README)."
        )
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def main():
    cfg = load_config("benchmark_config.json")

    runs         = int(cfg.get("runs"))
    seeds        = cfg.get("seeds")
    iterations   = int(cfg.get("iterations"))
    eval_every   = int(cfg.get("eval_every"))
    enable_dream = bool(cfg.get("enable_dream"))
    enable_cfr   = bool(cfg.get("enable_cfr"))

    # DREAM params
    dream_cfg       = cfg.get("dream", {}) or {}
    trajs_per_iter  = int(dream_cfg.get("trajs_per_iter"))
    batch_size      = int(dream_cfg.get("batch_size"))
    num_train_steps = int(dream_cfg.get("num_train_steps"))

    # Seeds
    if seeds is not None:
        if len(seeds) != runs:
            raise ValueError("Length of 'seeds' in JSON must equal 'runs'.")
    else:
        base = [101, 202, 303, 404, 505, 606, 707, 808, 909]
        if runs > len(base):
            extra = [1000 + 11*i for i in range(runs - len(base))]
            seeds = base + extra
        else:
            seeds = base[:runs]

    # Output folder
    output_root   = cfg.get("output_root", "results")
    benchmark_tag = cfg.get("benchmark_name", "benchmark")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(output_root, f"{benchmark_tag}_{stamp}")
    ensure_dir(root)
    print(f"[INFO] Writing everything under: {root}")

    # ----------------
    # DREAM runs
    # ----------------
    if enable_dream:
        for i, seed in enumerate(seeds, 1):
            print(f"\n==== DREAM run {i}/{runs} (seed={seed}) ====")
            iters, exploits = run_dream_once(
                seed=seed,
                iterations=iterations,
                eval_every=eval_every,
                trajs_per_iter=trajs_per_iter,
                batch_size=batch_size,
                num_train_steps=num_train_steps
            )
            out = os.path.join(root, f"DREAM_seed{seed}_run{i}.csv")
            write_run_csv(out, iters, exploits)

    # ----------------
    # CFR runs
    # ----------------
    if enable_cfr:
        for i, seed in enumerate(seeds, 1):
            print(f"\n==== CFR run {i}/{runs} (seed={seed}) ====")
            iters, exploits = run_cfr_once(
                seed=seed,
                iterations=iterations,
                eval_every=eval_every
            )
            out = os.path.join(root, f"CFR_seed{seed}_run{i}.csv")
            write_run_csv(out, iters, exploits)

    # ----------------
    # Aggregate & plot from output path
    # ----------------
    aggregate_and_plot_from_output(root)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"\nTotal wall time: {t1 - t0:.2f} s")
