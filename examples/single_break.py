"""Paper-replication single-break example."""

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
import numpy as np

from fmbqml import SingleBreakQML

SEED_SINGLE_BREAK = 2024
SEED_SINGLE_BREAK_CV = 12345

def replication_single_break(n_sim=5000):
    """Replicate Sections 4.1 and 4.2 of the fmbqml paper.

    This function contains three parts:
    1. a Type 1 single-break example;
    2. direct breakpoint estimation when a break is assumed to exist;
    3. a no-break control example constructed from the same simulated
       factors, loadings, and errors; and
    4. the LR profile reported as Figure 1.
    """
    rng = np.random.default_rng(SEED_SINGLE_BREAK)

    T = 300
    N = 100
    r0 = 3
    rho = 0.5
    breakpoint = 100

    print("=" * 70)
    print("Sections 4.1--4.2: Single-break simulation and LR profile")
    print("=" * 70)
    print(f"T = {T}, N = {N}, r0 = {r0}, rho = {rho}")
    print(f"True breakpoint = {breakpoint}")
    print(f"Simulation seed = {SEED_SINGLE_BREAK}")
    print(f"LR critical-value seed = {SEED_SINGLE_BREAK_CV}")

    # Generate AR(1) factors.
    u = rng.normal(0, np.sqrt(1 - rho**2), (T, r0))
    F = np.zeros((T, r0))
    F[0, :] = rng.normal(0, 1, r0)

    for t in range(1, T):
        F[t, :] = rho * F[t - 1, :] + u[t, :]

    # Generate loadings and idiosyncratic errors.
    Lambda1 = rng.normal(0, 1, (N, r0))
    Lambda2 = Lambda1 + rng.normal(0, 1, (N, r0))
    E = rng.normal(0, np.sqrt(r0), (T, N))

    # --------------------------------------------------------------
    # Alternative case: Type 1 break at breakpoint.
    # --------------------------------------------------------------
    X = np.zeros((T, N))
    X[:breakpoint, :] = F[:breakpoint, :] @ Lambda1.T + E[:breakpoint, :]
    X[breakpoint:, :] = F[breakpoint:, :] @ Lambda2.T + E[breakpoint:, :]

    print(f"Generated break-case data matrix X with shape {X.shape}")

    model = SingleBreakQML(
        X,
        trim_ratio=0.3,
        max_factors=10,
        factor_criterion="IC2"
    )

    print("\nRunning LR test for the break-case sample...")

    result_lr = model.lr_test(
        alpha=0.01,
        classify=True,
        verbose=True,
        n_sim=n_sim,
        random_state=SEED_SINGLE_BREAK_CV
    )

    # ============================================================
    # Save LR profile figure without opening a GUI window
    # ============================================================

    fig, ax = model.plot_lr_profile(
        true_break=breakpoint,
        show=False
    )
    plt.close(fig)

    print("\nRunning breakpoint estimation for the break-case sample...")

    result_qml = model.estimate_breakpoint(
        classify=True,
        verbose=True
    )

    # ============================================================
    # Save QML objective profile figure without opening a GUI window
    # ============================================================

    fig, ax = model.plot_qml_profile(
        true_break=breakpoint,
        show=False
    )
    plt.close(fig)
    # --------------------------------------------------------------
    # Null case: no structural break.
    # Reconstruct X_null explicitly and initialize a new model.
    # This matches the no-break control example in the paper.
    # --------------------------------------------------------------
    print("\nRunning no-break control example...")

    Lambda2_null = Lambda1.copy()

    X_null = np.zeros((T, N))
    X_null[:breakpoint, :] = F[:breakpoint, :] @ Lambda1.T + E[:breakpoint, :]
    X_null[breakpoint:, :] = F[breakpoint:, :] @ Lambda2_null.T + E[breakpoint:, :]

    print(f"Generated no-break data matrix X_null with shape {X_null.shape}")

    null_model = SingleBreakQML(
        X_null,
        trim_ratio=0.3,
        max_factors=10,
        factor_criterion="IC2"
    )

    null_result = null_model.lr_test(
        alpha=0.01,
        classify=False,
        verbose=True,
        n_sim=n_sim,
        random_state=SEED_SINGLE_BREAK_CV
    )

    fig, ax = null_model.plot_lr_profile(show=False)
    plt.close(fig)

    print("\nKey replication results:")
    print("Break-case reject null:", result_lr.get("reject_null"))
    print("Break-case estimated break:", result_lr.get("break_point"))
    print("No-break reject null:", null_result.get("reject_null"))
    print("No-break estimated break:", null_result.get("break_point"))

    return result_lr, result_qml, null_result


if __name__ == "__main__":
    replication_single_break()
