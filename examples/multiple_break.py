"""Multiple-break estimation and joint LR testing with fmbqml.

Demonstrates:
- Joint estimation of the number and locations of breakpoints
- Information-criterion (IC) path inspection
- Joint sup-LR testing

Related paper section: Section 5.1.
"""

# Standalone example corresponding to the tutorial.
# Requires fmbqml and numpy.
# Run: python examples/multiple_break.py

import numpy as np

from fmbqml import MultiBreakQML

print("\n" + "=" * 72)
print("Multiple-break estimation and joint LR testing with fmbqml")
print("=" * 72)

# 5.1. Generate the multiple-break panel
print("\n[1/5] Generating the multiple-break panel...")

# Use the fixed random stream used for this simulation example.
rng = np.random.RandomState(42)
T, N, r0 = 600, 100, 3
rho = 0.7
true_breakpoints = [150, 300, 450]

# Factor innovations have variance I; the initial factor has stationary variance 1/(1-rho^2).
u = rng.standard_normal((T, r0))
F = np.zeros((T, r0))
F[0, :] = u[0, :] / np.sqrt(1.0 - rho**2)
for t in range(1, T):
    F[t, :] = rho * F[t - 1, :] + u[t, :]

# Idiosyncratic errors are independent standard normal across time and units.
E = rng.standard_normal((T, N))
loadings = [
    rng.normal(scale=np.sqrt(1.0 / r0), size=(N, r0))
    for _ in range(4)
]
boundaries = [0, *true_breakpoints, T]
X = np.empty((T, N))
for regime, loading in enumerate(loadings):
    start, end = boundaries[regime:regime + 2]
    X[start:end, :] = F[start:end, :] @ loading.T + E[start:end, :]

# 5.1. Jointly estimate the number and locations of breakpoints
print("\n[2/5] Jointly estimating the number and locations of breakpoints...")

multi_model = MultiBreakQML(
    X,
    max_break=5,
    max_factors=15,
    factor_criterion="IC1",
    trim_ratio=0.10,
)

# Display the model configuration used for joint estimation and testing.
print("Model parameters:")
print(multi_model.get_parameters())

joint_results = multi_model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)
print(joint_results.summary())

# 5.1. Extract the joint-estimation results

print("\nJoint-estimation results:")
print(f"  Number of breaks : {joint_results.n_breaks}")
print(f"  Breakpoints      : {joint_results.breakpoints}")
print(f"  Break types      : {joint_results.break_types}")

# 5.1. Display the information-criterion path
print("\n[3/5] Displaying the information-criterion path...")

print("Information-criterion values across candidate break counts:")
print(joint_results.format_ic_path_table())

# 5.1. Test no breaks against exactly three breaks
print("\n[4/5] Performing the joint sup-LR test for three breaks...")


joint_lr_result = multi_model.joint_sup_lr_test(
    n_breaks=3,
    alpha=0.05,
    n_sim=1000,
    random_state=123,
)
print(joint_lr_result.summary())

# Appendix B. Figure 5: Multiple-break IC and QML-loss profiles
print("\n[5/5] Displaying the multiple-break diagnostic profiles...")

joint_results.plot_mqml_profile(show=True)

