"""Multiple-break example for fmbqml."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from fmbqml import MultiBreakQML


def run_multiple_break_example(n_sim=1000):
    """Run a simulated multiple-break example with joint estimation and testing."""
    rng = np.random.RandomState(42)
    T, N, r0 = 600, 100, 3
    rho = 0.7
    true_breakpoints = [150, 300, 450]

    print("Multiple-break example")
    print(f"T = {T}, N = {N}, r0 = {r0}, rho = {rho}")
    print(f"True breakpoints = {true_breakpoints}")

    # Generate the factor process.
    u = rng.standard_normal((T, r0))
    F = np.zeros((T, r0))
    F[0, :] = u[0, :] / np.sqrt(1.0 - rho**2)
    for t in range(1, T):
        F[t, :] = rho * F[t - 1, :] + u[t, :]

    # Generate idiosyncratic errors.
    E = rng.standard_normal((T, N))

    # Generate regime-specific loading matrices and construct the panel.
    loadings = [
        rng.normal(scale=np.sqrt(1.0 / r0), size=(N, r0))
        for _ in range(4)
    ]
    boundaries = [0, *true_breakpoints, T]
    X = np.empty((T, N))
    for regime, loading in enumerate(loadings):
        start, end = boundaries[regime:regime + 2]
        X[start:end, :] = F[start:end, :] @ loading.T + E[start:end, :]

    print(f"Generated data matrix X with shape {X.shape}")

    model = MultiBreakQML(
        X,
        max_break=5,
        max_factors=15,
        factor_criterion="IC1",
        trim_ratio=0.10,
    )

    print("\nJoint breakpoint estimation")
    joint_results = model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
    )
    print(joint_results.summary())

    print("\nSelected joint-estimation results")
    print(f"Selected number of breaks: {joint_results.n_breaks}")
    print(f"Estimated breakpoints: {joint_results.breakpoints}")
    print(f"Breakpoint types: {joint_results.break_types}")

    print("\nInformation-criterion path")
    print(joint_results.format_ic_path_table())

    output_dir = Path(__file__).resolve().parent / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "multiple_break_mqml_profile.png"
    fig, _ = joint_results.plot_mqml_profile(save_path=figure_path, show=False)
    plt.close(fig)
    print(f"MQML profile saved to:\n{figure_path}")

    print("\nJoint sup-LR test")
    joint_lr_result = model.joint_sup_lr_test(
        n_breaks=3,
        alpha=0.05,
        n_sim=n_sim,
        random_state=123,
        classify=True,
    )
    print(joint_lr_result.summary())

    print("\nSelected joint sup-LR results")
    print(f"LR statistic: {joint_lr_result.test_statistic:.4f}")
    print(f"Critical value: {joint_lr_result.critical_value:.4f}")
    print(f"Reject H0: {joint_lr_result.reject_null}")
    print(f"Candidate breakpoints under H1: {joint_lr_result.candidate_breakpoints}")
    print(f"Accepted breakpoints: {joint_lr_result.breakpoints}")

    return {
        "joint_results": joint_results,
        "joint_lr_result": joint_lr_result,
    }


if __name__ == "__main__":
    run_multiple_break_example()
