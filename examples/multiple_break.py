"""Paper-replication multiple-break example."""

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from fmbqml import MultiBreakQML

SEED_MULTIPLE_BREAK = 42

def simulate_no_break_factor_data(
        T=600,
        N=100,
        r0=3,
        rho=0.7,
        alpha=0.0,
        beta=0.0,
        seed=SEED_MULTIPLE_BREAK
):
    """Generate a factor panel with no loading breaks."""
    if not -1 < rho < 1 or not -1 < alpha < 1:
        raise ValueError("rho and alpha must lie between -1 and 1.")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1).")

    rng = np.random.default_rng(seed)

    factor_shocks = rng.standard_normal((T, r0))
    factors = np.zeros((T, r0))
    factors[0, :] = factor_shocks[0, :] / np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        factors[t, :] = rho * factors[t - 1, :] + factor_shocks[t, :]

    distances = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    covariance = beta ** distances
    covariance_sqrt = sqrtm(covariance + 1e-8 * np.eye(N)).real
    correlated_shocks = rng.standard_normal((T, N)) @ covariance_sqrt

    errors = np.zeros((T, N))
    errors[0, :] = correlated_shocks[0, :] / np.sqrt(1.0 - alpha ** 2)
    for t in range(1, T):
        errors[t, :] = alpha * errors[t - 1, :] + correlated_shocks[t, :]

    loading = rng.normal(0.0, np.sqrt(1.0 / r0), size=(N, r0))
    X = factors @ loading.T + errors

    return X, factors, errors, loading


def _print_prespecified_lr_result(result, true_breakpoints):
    """Print the decision and breakpoint estimates in a consistent format."""
    reject = bool(result["reject_null"])
    estimated = result.get("breakpoints", result.get("estimated_breakpoints"))

    print("Reject the no-break null:", reject)
    print("Structural breaks detected:", reject)
    print("Estimated breakpoints:", estimated if reject else "None")
    print("True breakpoints:", true_breakpoints)

def replication_multiple_breaks(
        lr_n_sim=1000,
        lr_random_state=123
):
    """Replicate Sections 5.2--5.4 of the fmbqml paper.

    This function contains:
    1. the multiple-break simulation design from Section 5.2;
    2. the joint estimation results and IC path from Section 5.3 (Table 9);
    3. the joint sup-LR test from Section 5.4; and
    4. a no-break control used as an additional diagnostic.
    """
    # ============================================================
    # 1. Simulation settings
    # ============================================================

    r0 = 3
    N = 100
    T = 600
    m = 3

    t1 = round(T / (m + 1))
    t2 = round(T / (m + 1) * 2)
    t3 = round(T / (m + 1) * 3)

    print("\n" + "=" * 70)
    print("Sections 5.2--5.4: Multiple-break simulation and inference")
    print("=" * 70)
    print(f"T = {T}, N = {N}, r0 = {r0}, m = {m}")
    print(f"True breakpoints: {t1}, {t2}, {t3}")
    print(f"Simulation seed = {SEED_MULTIPLE_BREAK}")

    rho = 0.7
    alpha = 0
    beta = 0
    trim_ratio = 0.1

    np.random.seed(SEED_MULTIPLE_BREAK)

    # ============================================================
    # 2. Generate AR(1) factors
    # ============================================================

    u = np.random.randn(T, r0)

    F = np.zeros((T, r0))
    F[0, :] = np.sqrt(1 / (1 - rho ** 2)) * u[0, :]

    for t in range(1, T):
        F[t, :] = rho * F[t - 1, :] + u[t, :]

    # ============================================================
    # 3. Generate idiosyncratic errors
    # ============================================================

    E = np.zeros((T, N))
    w = np.random.randn(T, N)

    sigma = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            sigma[i, j] = beta ** abs(i - j)

    sigma_sqrt = sqrtm(sigma + 1e-8 * np.eye(N))
    v = w @ sigma_sqrt.real

    E[0, :] = v[0, :] / np.sqrt(1 - alpha ** 2)

    for t in range(1, T):
        E[t, :] = alpha * E[t - 1, :] + v[t, :]

    e = E.T

    # ============================================================
    # 4. Generate loading matrices for four regimes
    # ============================================================

    b = 0

    alpha1 = np.random.normal(0.5 * b, np.sqrt(1 / r0), (N, r0))
    alpha2 = np.random.normal(b, np.sqrt(1 / r0), (N, r0))
    alpha3 = np.random.normal(1.5 * b, np.sqrt(1 / r0), (N, r0))
    alpha4 = np.random.normal(2.0 * b, np.sqrt(1 / r0), (N, r0))

    X1 = alpha1 @ F[:t1, :].T + e[:, :t1]
    X2 = alpha2 @ F[t1:t2, :].T + e[:, t1:t2]
    X3 = alpha3 @ F[t2:t3, :].T + e[:, t2:t3]
    X4 = alpha4 @ F[t3:, :].T + e[:, t3:]

    X = np.hstack([X1, X2, X3, X4]).T

    print(f"Generated data matrix X with shape {X.shape}")

    # ============================================================
    # 6. Initialize MultiBreakQML model
    # ============================================================

    model = MultiBreakQML(
        X,
        max_break=5,
        max_factors=15,
        factor_criterion=0,
        trim_ratio=trim_ratio,
    )

    # ============================================================
    # 7. Joint breakpoint estimation
    # ============================================================

    print("\nRunning joint breakpoint estimation...")

    res_joint = model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
        verbose=True
    )

    fig, ax = model.plot_mqml_profile(show=False)
    plt.close(fig)

    # ============================================================
    # 8. Export information criterion path
    # ============================================================

    print("\nPreparing information criterion path...")

    if "ic_path" not in res_joint:
        raise KeyError(
            "res_joint does not contain 'ic_path'. "
            "Please make sure joint_estimation() returns ic_path "
            "and estimate_breaks_jointly() saves it into joint_results."
        )

    ic_table = pd.DataFrame(res_joint["ic_path"])

    if ic_table.empty:
        raise ValueError(
            "The IC path table is empty. Please check joint_estimation()."
        )

    def format_breakpoints(x):
        """Convert breakpoint lists to clean strings for CSV and LaTeX output."""
        if x is None:
            return "[]"

        if isinstance(x, str):
            return x

        if isinstance(x, (list, tuple, np.ndarray)):
            x_list = list(np.asarray(x).flatten())
            if len(x_list) == 0:
                return "[]"
            return "[" + ", ".join(str(int(v)) for v in x_list) + "]"

        try:
            x_list = list(x)
            if len(x_list) == 0:
                return "[]"
            return "[" + ", ".join(str(int(v)) for v in x_list) + "]"
        except TypeError:
            return str(x)

    if "estimated_breakpoints" in ic_table.columns:
        ic_table["estimated_breakpoints"] = ic_table["estimated_breakpoints"].apply(
            format_breakpoints
        )

    ic_table_for_paper = ic_table.rename(
        columns={
            "m": r"$m$",
            "IC": r"$IC(m)$",
            "loss": "QML loss",
            "estimated_breakpoints": "Estimated breakpoints"
        }
    )

    print("\nInformation criterion path:")
    print(ic_table_for_paper.to_string(index=False))

    # ============================================================
    # 9. joint_sup_lr_test example
    # ============================================================,

    print("\nRunning MultiBreakQML.joint_sup_lr_test example...")

    # Use the paper's maximum-factor setting for the joint sup-LR example.
    lr_model = MultiBreakQML(
        X,
        max_break=5,
        max_factors=15,
        factor_criterion=0,
        trim_ratio=trim_ratio,
    )
    res_lr = lr_model.joint_sup_lr_test(
        n_breaks=m,
        alpha=0.05,
        n_sim=lr_n_sim,
        random_state=lr_random_state,
        classify=True,
        verbose=True
    )
    _print_prespecified_lr_result(res_lr, [t1, t2, t3])

    fig, ax = lr_model.plot_joint_lr_profile(
        alpha=0.05,
        n_sim=lr_n_sim,
        random_state=lr_random_state,
        show=False,
    )
    plt.close(fig)

    res_joint["joint_sup_lr_test"] = res_lr

    # --------------------------------------------------------------
    # No-break control: run the same multiple-break LR test on a
    # panel generated with one fixed loading matrix across the sample.
    # --------------------------------------------------------------
    print("\nRunning multiple-break no-break control example...")

    X_null, _, _, _ = simulate_no_break_factor_data(
        T=T,
        N=N,
        r0=r0,
        rho=rho,
        alpha=alpha,
        beta=beta,
        seed=SEED_MULTIPLE_BREAK,
    )
    print(f"Generated no-break data matrix X_null with shape {X_null.shape}")

    null_model = MultiBreakQML(
        X_null,
        max_break=5,
        max_factors=15,
        factor_criterion=0,
        trim_ratio=trim_ratio,
    )
    res_lr_null = null_model.joint_sup_lr_test(
        n_breaks=m,
        alpha=0.05,
        n_sim=lr_n_sim,
        random_state=lr_random_state,
        classify=False,
        verbose=True
    )
    _print_prespecified_lr_result(res_lr_null, [])

    fig, ax = null_model.plot_joint_lr_profile(
        alpha=0.05,
        n_sim=lr_n_sim,
        random_state=lr_random_state,
        show=False,
    )
    plt.close(fig)

    res_joint["joint_sup_lr_test_no_break"] = res_lr_null

    # ============================================================
    # 10. Final summary
    # ============================================================

    print("\nFinal multiple-break replication results:")
    print("True breakpoints:", [t1, t2, t3])
    print("Joint estimated breakpoints:", res_joint.get("breakpoints"))
    print("Joint estimated number of breaks:", res_joint.get("n_breaks"))
    print("joint_sup_lr_test rejects no-break null:", res_lr.get("reject_null"))
    print(
        "joint_sup_lr_test estimated breakpoints:",
        res_lr.get("breakpoints", res_lr.get("estimated_breakpoints"))
    )
    print(
        "No-break joint_sup_lr_test rejects no-break null:",
        res_lr_null.get("reject_null")
    )
    print(
        "No-break joint_sup_lr_test estimated breakpoints:",
        res_lr_null.get("breakpoints") if res_lr_null.get("reject_null") else []
    )

    return res_joint, ic_table_for_paper


if __name__ == "__main__":
    replication_multiple_breaks()
