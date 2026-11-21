"""Single-break estimation and testing with fmbqml.

Demonstrates:
- Single-break likelihood-ratio (LR) testing
- QML breakpoint estimation
- A no-break control

Related paper section: Section 4.1.
"""

# Standalone example corresponding to the tutorial.
# Requires fmbqml and numpy.
# Run: python examples/single_break.py

import numpy as np

from fmbqml import SingleBreakQML

print("\n" + "=" * 72)
print("Single-break estimation and testing with fmbqml")
print("=" * 72)

# 4.1. Generate the single-break panel
print("\n[1/7] Generating the single-break panel...")

rng = np.random.default_rng(2024)
T, N, r0 = 300, 100, 3
rho = 0.5
true_breakpoint = 100

# The factor process is stationary: Var(f_1) = I and Var(u_t) = (1-rho^2)I.
u = rng.normal(0.0, np.sqrt(1.0 - rho**2), size=(T, r0))
F = np.zeros((T, r0))
F[0, :] = rng.normal(0.0, 1.0, size=r0)
for t in range(1, T):
    F[t, :] = rho * F[t - 1, :] + u[t, :]

# Generate the pre-break and post-break loading matrices and idiosyncratic errors.
Lambda1 = rng.normal(0.0, 1.0, size=(N, r0))
Lambda2 = Lambda1 + rng.normal(0.0, 1.0, size=(N, r0))
E = rng.normal(0.0, np.sqrt(r0), size=(T, N))

# The observed panel follows the one-break factor model.
X = np.empty((T, N))
X[:true_breakpoint, :] = (
    F[:true_breakpoint, :] @ Lambda1.T + E[:true_breakpoint, :]
)
X[true_breakpoint:, :] = (
    F[true_breakpoint:, :] @ Lambda2.T + E[true_breakpoint:, :]
)

# 4.1. Initialize `SingleBreakQML` and perform the LR test
print("\n[2/7] Performing the single-break LR test (Monte Carlo critical value)...")

model = SingleBreakQML(
    X,
    trim_ratio=0.30,
    max_factors=10,
    factor_criterion="IC2",
)

# Display the model configuration used for the LR test.
print("Model parameters:")
print(model.get_parameters())

lr_result = model.lr_test(
    alpha=0.01,
    classify=True,
    n_sim=5000,
    random_state=12345,
)
print(lr_result.summary())

# 4.1. Extract the LR-test results

print("\nLR-test results:")
print(f"  Test statistic : {lr_result.test_statistic:.4f}")
print(f"  Critical value : {lr_result.critical_value:.4f}")
print(f"  Reject null    : {lr_result.reject_null}")
print(f"  Breakpoint     : {lr_result.breakpoint}")
print(f"  Break type     : {lr_result.break_type}")


# 4.1. Figure 1: LR profile
print("\n[3/7] Displaying the LR profile...")

lr_result.plot_lr_profile(true_break=true_breakpoint, show=True)

# 4.1. Direct QML breakpoint estimation
print("\n[4/7] Estimating the breakpoint directly using QML...")

qml_result = model.estimate_breakpoint(classify=True)
print(qml_result.summary())

# 4.1. Construct the no-break control panel
print("\n[5/7] Constructing the no-break control panel...")

# Keep the same factors, first-regime loadings, and idiosyncratic errors.
X_null = F @ Lambda1.T + E

# 4.1. Test the no-break control panel
print("\n[6/7] Testing the no-break control panel...")

null_model = SingleBreakQML(
    X_null,
    trim_ratio=0.30,
    max_factors=10,
    factor_criterion="IC2",
)
null_result = null_model.lr_test(
    alpha=0.01,
    classify=False,
    n_sim=5000,
    random_state=12345,
)
print(null_result.summary())

# Appendix A. Figure 3: QML objective profile for the single-break example
print("\n[7/7] Displaying the appendix diagnostic profiles...")

qml_result.plot_qml_profile(
    true_break=true_breakpoint,
    show=True,
)

# Appendix A. Figure 4: LR profile for the no-break control

null_result.plot_lr_profile(show=True)

