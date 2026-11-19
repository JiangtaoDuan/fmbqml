"""Single-break example for fmbqml."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from fmbqml import SingleBreakQML


def run_single_break_example(n_sim=5000):
    """Run a simulated single-break example and a no-break control."""
    rng = np.random.default_rng(2024)
    T, N, r0 = 300, 100, 3
    rho = 0.5
    true_breakpoint = 100

    print("Single-break example")
    print(f"T = {T}, N = {N}, r0 = {r0}, rho = {rho}")
    print(f"True breakpoint = {true_breakpoint}")

    # Generate the factor process.
    u = rng.normal(0.0, np.sqrt(1.0 - rho**2), size=(T, r0))
    F = np.zeros((T, r0))
    F[0, :] = rng.normal(0.0, 1.0, size=r0)
    for t in range(1, T):
        F[t, :] = rho * F[t - 1, :] + u[t, :]

    # Generate loadings and idiosyncratic errors.
    Lambda1 = rng.normal(0.0, 1.0, size=(N, r0))
    Lambda2 = Lambda1 + rng.normal(0.0, 1.0, size=(N, r0))
    E = rng.normal(0.0, np.sqrt(r0), size=(T, N))

    # Construct the panel with one loading break.
    X = np.empty((T, N))
    X[:true_breakpoint, :] = (
        F[:true_breakpoint, :] @ Lambda1.T + E[:true_breakpoint, :]
    )
    X[true_breakpoint:, :] = (
        F[true_breakpoint:, :] @ Lambda2.T + E[true_breakpoint:, :]
    )
    print(f"Generated data matrix X with shape {X.shape}")

    model = SingleBreakQML(
        X,
        trim_ratio=0.30,
        max_factors=10,
        factor_criterion="IC2",
    )

    print("\nLikelihood-ratio test")
    lr_result = model.lr_test(
        alpha=0.01,
        classify=True,
        n_sim=n_sim,
        random_state=12345,
    )
    print(lr_result.summary())

    print("\nSelected LR-test results")
    print(f"LR statistic: {lr_result.test_statistic:.4f}")
    print(f"Critical value: {lr_result.critical_value:.4f}")
    print(f"Reject H0: {lr_result.reject_null}")
    print(f"Accepted breakpoint: {lr_result.breakpoint}")
    print(f"Breakpoint type: {lr_result.break_type}")

    output_dir = Path(__file__).resolve().parent / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "single_break_lr_profile.png"
    fig, _ = lr_result.plot_lr_profile(
        true_break=true_breakpoint,
        save_path=figure_path,
        show=False,
    )
    plt.close(fig)
    print(f"LR profile saved to: {figure_path}")

    print("\nDirect QML breakpoint estimation")
    qml_result = model.estimate_breakpoint(classify=True)
    print(qml_result.summary())

    figure_path = output_dir / "single_break_qml_profile.png"
    fig, _ = qml_result.plot_qml_profile(
        true_break=true_breakpoint,
        save_path=figure_path,
        show=False,
    )
    plt.close(fig)
    print(f"QML profile saved to: {figure_path}")

    # Construct a no-break panel using the same factors, first-regime
    # loadings, and idiosyncratic errors.
    X_null = F @ Lambda1.T + E
    print(f"Generated no-break data matrix X_null with shape {X_null.shape}")

    null_model = SingleBreakQML(
        X_null,
        trim_ratio=0.30,
        max_factors=10,
        factor_criterion="IC2",
    )
    print("\nNo-break control")
    null_result = null_model.lr_test(
        alpha=0.01,
        classify=False,
        n_sim=n_sim,
        random_state=12345,
    )
    print(null_result.summary())

    figure_path = output_dir / "no_break_lr_profile.png"
    fig, _ = null_result.plot_lr_profile(save_path=figure_path, show=False)
    plt.close(fig)
    print(f"LR profile saved to: {figure_path}")

    return {
        "lr_result": lr_result,
        "qml_result": qml_result,
        "null_result": null_result,
    }


if __name__ == "__main__":
    run_single_break_example()
