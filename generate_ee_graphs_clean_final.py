# generate_ee_graphs_clean_final.py
#
# Generates the same five EE figures as the cleaned graph script.
#
# Important behaviour preserved:
#   - Same input files: results/results.csv and results/histories.json
#   - Same five PDF output filenames
#   - Same plotted data, labels, titles, markers, line styles, and grid settings
#   - Bisection continues to use the stored history, exactly as before
#   - Regula Falsi histories are regenerated directly in memory at full float
#     precision so the fitted slopes are not distorted by JSON rounding
#   - Regula Falsi graphs show only experimental points, experimental
#     least-squares trendlines, and experimental slopes in the legend
#   - Regula Falsi prints one experimental and one theoretical slope per run
#   - No graph windows are opened; every figure is saved and immediately closed

from methods import f, fp, markley_kepler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

import matplotlib
from matplotlib.ticker import MultipleLocator

# Force a non-interactive backend so rerunning this file never opens windows.
# This must be set before importing matplotlib.pyplot.
matplotlib.use("Agg")


# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "results.csv"
HISTORIES_JSON = RESULTS_DIR / "histories.json"

TOL = 1e-8
MAX_ITER = 100
SELECTED_E = [0.05, 0.50, 0.95]
METHODS = ["Bisection", "Regula Falsi", "Newton-Raphson"]


# ============================================================
# COMMON HELPERS
# ============================================================

def save_and_close(output_file, *, bbox_inches=None):
    """Save the current figure exactly once, then close it."""
    plt.tight_layout()

    if bbox_inches is None:
        plt.savefig(output_file)
    else:
        plt.savefig(output_file, bbox_inches=bbox_inches)

    plt.close()


def load_inputs():
    """Load the same stored experiment data used by the original graphs.py."""
    df = pd.read_csv(RESULTS_CSV)

    with open(HISTORIES_JSON, "r", encoding="utf-8") as file:
        histories = json.load(file)

    return df, histories


# ============================================================
# ITERATION COUNT VS ECCENTRICITY
# ============================================================

def plot_iteration_counts(df, M_label, title_M, output_file):
    """Plot iteration count against eccentricity with points connected for visual guidance."""
    subset_M = df[df["M_label"] == M_label]

    # No figsize is supplied, preserving Matplotlib's original default size.
    plt.figure()

    for method in METHODS:
        subset = subset_M[subset_M["method"] == method]

        plt.plot(
            subset["e"],
            subset["iterations"],
            marker="o",
            linewidth=0.8,
            markersize=4,
            label=method,
        )

    plt.xlabel("Eccentricity, e")
    plt.ylabel("Iteration count, N")
    plt.title(f"Iteration Count vs Eccentricity for M = {title_M}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close(output_file)


# ============================================================
# BISECTION CONVERGENCE HISTORY
# ============================================================

def plot_bisection_history(histories, M, M_label, title_M, output_file):
    """
    Plot the Bisection convergence history for one M value.

    This continues to use results/histories.json so the plotted data and
    behaviour remain identical to the existing Bisection graph.
    """
    method = "Bisection"

    theoretical_gradient = -np.log10(2)

    plt.figure(figsize=(8, 5))

    for e in SELECTED_E:
        record = next(
            item
            for item in histories
            if item["method"] == method
            and item["M_label"] == M_label
            and abs(item["e"] - e) < 1e-10
        )

        history = record["history"]
        markley_root = markley_kepler(e, M)

        errors = np.array(
            [abs(E_n - markley_root) for E_n in history],
            dtype=float,
        )

        mask = errors > 0
        errors = errors[mask]
        iterations = np.arange(1, len(history) + 1)[mask]
        log_errors = np.log10(errors)

        gradient, intercept = np.polyfit(
            iterations,
            log_errors,
            1,
        )

        plt.plot(
            iterations,
            log_errors,
            marker="o",
            markersize=4,
            linewidth=0.8,
            label=fr"$e={e:.2f}$, $m={gradient:.3f}$",
        )

        fitted_values = gradient * iterations + intercept

        plt.plot(
            iterations,
            fitted_values,
            linestyle="--",
            linewidth=1,
        )

    plt.xlabel("Iteration, n")
    plt.ylabel(
        r"$\log_{10}\!\left(|E_n-E_{\mathrm{Markley}}|\right)$"
    )
    plt.title(rf"Bisection Convergence History for $M={title_M}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    save_and_close(output_file)

    print(
        f"Theoretical Bisection gradient for M = {M_label}: "
        f"{theoretical_gradient:.4f}"
    )


# ============================================================
# FULL-PRECISION REGULA FALSI HISTORY
# ============================================================

def regula_falsi_history(e, M, tol=TOL, max_iter=MAX_ITER):
    """
    Run classical Regula Falsi directly in memory.

    Initial bracket: [M, M + e]
    Stopping criterion: |f(E_n)| < tol

    Regenerating the iterates here avoids the old precision loss caused by
    storing very small-error histories at insufficient decimal precision.
    """
    a = M
    b = M + e

    fa = f(a, e, M)
    fb = f(b, e, M)

    history = []

    for _ in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        history.append(c)

        fc = f(c, e, M)

        if abs(fc) < tol:
            break

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return np.array(history, dtype=float)


# ============================================================
# REGULA FALSI THEORETICAL SLOPE
# ============================================================

def get_retained_endpoint(e, M):
    """Return the endpoint retained after the first Regula Falsi step."""
    a = M
    b = M + e

    fa = f(a, e, M)
    fb = f(b, e, M)

    c = b - fb * (b - a) / (fb - fa)
    fc = f(c, e, M)

    if fa * fc < 0:
        return a
    return b


def theoretical_regula_slope(e, M):
    """
    Return the asymptotic theoretical slope used in the earlier comparison.

    The returned value is the gradient on the log10(error) vs iteration graph.
    Only the slope is exposed in the output; no extra convergence-factor
    notation is printed.
    """
    root = markley_kepler(e, M)
    q = get_retained_endpoint(e, M)

    fq = f(q, e, M)
    fp_root = fp(root, e)

    factor = abs(1.0 - ((q - root) * fp_root) / fq)
    return np.log10(factor)


def analyse_regula_case(e, M):
    """Return full-precision Regula Falsi history and both run slopes."""
    history = regula_falsi_history(e, M)
    root = markley_kepler(e, M)

    errors = np.abs(history - root)
    mask = errors > 0

    errors = errors[mask]
    iterations = np.arange(1, len(history) + 1)[mask]
    log_errors = np.log10(errors)

    if len(iterations) < 2:
        raise ValueError(
            f"Not enough non-zero error points to fit a trendline "
            f"for e={e}, M={M}."
        )

    slope_exp, intercept_exp = np.polyfit(
        iterations,
        log_errors,
        1,
    )

    slope_theory = theoretical_regula_slope(e, M)

    return {
        "iterations": iterations,
        "errors": errors,
        "log_errors": log_errors,
        "slope_exp": slope_exp,
        "intercept_exp": intercept_exp,
        "slope_theory": slope_theory,
    }


# ============================================================
# REGULA FALSI CONVERGENCE GRAPHS
# ============================================================

def print_regula_comparison(e, result):
    """Print exactly one experimental and one theoretical slope per run."""
    slope_exp = result["slope_exp"]
    slope_theory = result["slope_theory"]
    difference = slope_exp - slope_theory

    print(
        f"e = {e:.2f}: "
        f"experimental = {slope_exp:.4f}, "
        f"theoretical = {slope_theory:.4f}, "
        f"difference = {difference:+.4f}"
    )


def plot_regula_history(M, M_label, title_M, output_file):
    """
    Plot the corrected full-precision Regula Falsi convergence history.

    The graph itself contains only:
      - experimental convergence points,
      - the experimental least-squares trendline,
      - the experimental trendline slope in the legend.

    The terminal prints exactly one experimental slope and one theoretical
    asymptotic slope for each run. No extra factor is printed.
    """
    plt.figure(figsize=(8, 5))

    print()
    print(f"========== M = {M_label} ==========")

    for e in SELECTED_E:
        result = analyse_regula_case(e, M)

        iterations = result["iterations"]
        log_errors = result["log_errors"]
        slope_exp = result["slope_exp"]
        intercept_exp = result["intercept_exp"]

        plt.plot(
            iterations,
            log_errors,
            marker="o",
            markersize=4,
            linewidth=0.8,
            label=(
                fr"$e={e:.2f}$, "
                fr"$m={slope_exp:.3f}$"
            ),
        )

        fitted_exp = slope_exp * iterations + intercept_exp

        plt.plot(
            iterations,
            fitted_exp,
            linestyle="--",
            linewidth=1,
        )

        print_regula_comparison(e, result)

    plt.xlabel("Iteration, $n$")
    plt.ylabel(
        r"$\log_{10}\!\left(|E_n-E_{\mathrm{Markley}}|\right)$"
    )
    plt.title(
        rf"Regula Falsi Convergence History for $M={title_M}$"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close(output_file, bbox_inches="tight")

# ============================================================
# NEWTON-RAPHSON CONVERGENCE HISTORY
# ============================================================


def newton_history(e, M, tol=TOL, max_iter=MAX_ITER):
    """
    Run Newton-Raphson directly in memory at full float precision.

    Initial guess: E0 = M + e/2
    Stopping criterion: |f(E_n)| < tol

    The initial guess is stored as iteration n = 0.
    """
    E = M + e / 2
    history = [E]

    for _ in range(max_iter):
        E = E - f(E, e, M) / fp(E, e)
        history.append(E)

        if abs(f(E, e, M)) < tol:
            break

    return np.array(history, dtype=float)


def plot_newton_history(M, title_M, output_file):
    """
    Plot Newton-Raphson convergence as iteration number against
    log10 absolute error relative to Markley's reference solution.
    """
    plt.figure(figsize=(8, 5))

    max_n = 0

    for e in SELECTED_E:
        history = newton_history(e, M)
        markley_root = markley_kepler(e, M)

        errors = np.abs(history - markley_root)

        # log10(0) is undefined, so remove points where floating-point
        # agreement with the Markley solution makes the error exactly zero.
        mask = errors > 0

        errors = errors[mask]
        iterations = np.arange(len(history))[mask]
        log_errors = np.log10(errors)

        max_n = max(max_n, int(iterations.max()))

        plt.plot(
            iterations,
            log_errors,
            marker="o",
            markersize=4,
            linewidth=0.8,
            label=fr"$e={e:.2f}$",
        )

    plt.xlabel("Iteration, $n$")
    plt.ylabel(
        r"$\log_{10}\!\left(|E_n-E_{\mathrm{Markley}}|\right)$"
    )
    plt.title(
        rf"Newton-Raphson Convergence History for $M={title_M}$"
    )

    # Iterations are discrete, so show integer ticks only.
    plt.xticks(np.arange(0, max_n + 1, 1))

    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close(output_file, bbox_inches="tight")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Running graph generator: {Path(__file__).resolve()}")
    print("Regula Falsi output shows one experimental slope and one theoretical slope per run.")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, histories = load_inputs()

    # Same two iteration-count PDFs as before.
    plot_iteration_counts(
        df=df,
        M_label="0.1",
        title_M="0.1",
        output_file=RESULTS_DIR / "iterations_M01.pdf",
    )

    plot_iteration_counts(
        df=df,
        M_label="pi/2",
        title_M="pi/2",
        output_file=RESULTS_DIR / "iterations_Mpi2.pdf",
    )

    # Bisection convergence-history PDFs.
    plot_bisection_history(
        histories=histories,
        M=0.1,
        M_label="0.1",
        title_M="0.1",
        output_file=RESULTS_DIR / "bisection_convergence_M01.pdf",
    )

    plot_bisection_history(
        histories=histories,
        M=np.pi / 2,
        M_label="pi/2",
        title_M=r"\frac{\pi}{2}",
        output_file=RESULTS_DIR / "bisection_convergence_MPI2.pdf",
    )

    # Same Regula Falsi PDF filenames, using full-precision in-memory histories.
    plot_regula_history(
        M=0.1,
        M_label="0.1",
        title_M="0.1",
        output_file=RESULTS_DIR / "regula_history_M01.pdf",
    )

    plot_regula_history(
        M=np.pi / 2,
        M_label="pi/2",
        title_M=r"\frac{\pi}{2}",
        output_file=RESULTS_DIR / "regula_history_MPI2.pdf",
    )

    # Newton-Raphson convergence-history PDFs.
    plot_newton_history(
        M=0.1,
        title_M="0.1",
        output_file=RESULTS_DIR / "newton_history_M01.pdf",
    )

    plot_newton_history(
        M=np.pi / 2,
        title_M=r"\frac{\pi}{2}",
        output_file=RESULTS_DIR / "newton_history_MPI2.pdf",
    )

    print()
    print("All graphs saved. No graph windows were opened.")


if __name__ == "__main__":
    main()
