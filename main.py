import numpy as np
import pandas as pd
from pathlib import Path

from methods import (
    bisection,
    regula_falsi,
    newton_raphson,
    markley_kepler,
    f,
)


# Experimental parameters
eccentricities = np.arange(0.05, 1.00, 0.05)
mean_anomalies = [0.1, np.pi / 2]

results = []

Path("results").mkdir(exist_ok=True)


def make_result(
    e,
    M,
    method,
    root,
    iterations,
    history,
    markley_root,
):
    converged = root is not None

    if converged:
        residual = abs(f(root, e, M))
        absolute_error = abs(root - markley_root)
    else:
        residual = np.nan
        absolute_error = np.nan

    return {
        "e": e,
        "M": M,
        "M_label": "0.1" if M == 0.1 else "pi/2",
        "method": method,
        "root": root,
        "iterations": iterations,
        "residual": residual,
        "absolute_error": absolute_error,
        "converged": converged,
        "history": history,
    }


for M in mean_anomalies:
    for e in eccentricities:

        # Avoid floating-point labels such as 0.15000000000000002
        e = round(float(e), 2)

        # Benchmark solution
        markley_root = markley_kepler(e, M)

        # Run numerical methods
        bisection_root, bisection_iterations, bisection_history = bisection(
            e, M
        )
        regula_root, regula_iterations, regula_history = regula_falsi(
            e, M
        )
        newton_root, newton_iterations, newton_history = newton_raphson(
            e, M
        )

        # Store Bisection result
        results.append(
            make_result(
                e,
                M,
                "Bisection",
                bisection_root,
                bisection_iterations,
                bisection_history,
                markley_root,
            )
        )

        # Store Regula Falsi result
        results.append(
            make_result(
                e,
                M,
                "Regula Falsi",
                regula_root,
                regula_iterations,
                regula_history,
                markley_root,
            )
        )

        # Store Newton-Raphson result
        results.append(
            make_result(
                e,
                M,
                "Newton-Raphson",
                newton_root,
                newton_iterations,
                newton_history,
                markley_root,
            )
        )

        print(
            f"Completed e = {e:.2f}, "
            f"M = {'0.1' if M == 0.1 else 'pi/2'}"
        )


# Convert the collected rows into a table
results_df = pd.DataFrame(results)

# Save the main numerical results
results_df.drop(columns=["history"]).to_csv(
    "results/results.csv",
    index=False,
)

# Save iteration histories separately
results_df[
    ["e", "M", "M_label", "method", "history"]
].to_json(
    "results/histories.json",
    orient="records",
    indent=2,
)

print()
print("Experiment complete.")
print(f"Rows collected: {len(results_df)}")
print("Saved main results to results/results.csv")
print("Saved histories to results/histories.json")
