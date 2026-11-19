"""Standalone numerical verification.

Run with ``python -m tests.audit_numerical_verification``.
"""
import json
import numpy as np

from fmbqml.multiple_break.joint_testing import _run_joint_sup_lr, _normalize_factors
from fmbqml.statistical_tools.hac import hac_newey_west_94
from fmbqml.statistical_tools.critical_values import (
    simulate_single_sup_lr_critical_values,
)


def independent_hac_reference(x, cons, pw):
    """Independent NumPy transcription of the supplied HAC_NW94_new1.m.

    Only p_min=p_max=1 is used, so the VAR coefficients are computed by
    independent least squares rather than calling fmbqml.fit_var.
    This is *not* execution of the user's MATLAB program.
    """
    T, k = x.shape
    Y = x.copy()
    w = np.ones(k) if cons == 0 else np.r_[0., np.ones(k - 1)]
    if pw:
        lagged, dependent = Y[:-1], Y[1:]
        beta, *_ = np.linalg.lstsq(lagged, dependent, rcond=None)
        Betahat = beta.T
        Z = dependent - lagged @ beta
    else:
        Z = Y
    T = len(Z)
    n = int(np.floor(4 * (T / 100) ** (2 / 9)))
    s = []
    for v in range(n + 1):
        gamma = Z[:T-v].T @ Z[v:] / T
        s.append(float(w @ gamma @ w))
    s0 = 2 * sum(s) - s[0]
    sq = sum(2*v*s[v] for v in range(n + 1))
    rhat = 1.5 * 1.1447 * ((sq/s0)**2)**(1/3)
    ST = int(np.floor(rhat * T**(1/3)))
    phi = np.zeros((k, k))
    for lag in range(ST+1):
        G = Z[:T-lag].T @ Z[lag:] / T
        phi += (1 - lag/(ST+1)) * (G + G.T)
    gamma0 = Z.T @ Z / T
    V = phi - gamma0
    if pw:
        D = np.eye(k) - Betahat
        inv = np.linalg.inv(D)
        V = inv @ V @ inv.T
    return V


def main():
    rng = np.random.default_rng(541)
    T = 100
    z = rng.standard_normal((T, 2))
    x = np.empty_like(z)
    x[0] = z[0]
    for t in range(1, T):
        x[t] = np.array([[0.35, 0.02], [0.01, -0.12]]) @ x[t-1] + z[t]

    hac_diffs = {}
    for cons in (0, 1):
        for pw in (0, 1):
            got = hac_newey_west_94(x, np.ones(T), cons, 1, 4, pw, 1, 1, 1.5)
            expected = independent_hac_reference(x, cons, pw)
            hac_diffs[f"cons={cons},pw={pw}"] = float(np.max(np.abs(got - expected)))

    f = rng.standard_normal((90, 2))
    f = _normalize_factors(f)
    joint = _run_joint_sup_lr(
        f,
        n_breaks=1,
        trim_ratio=0.2,
        alpha=0.05,
        n_sim=6,
        grid_size=20,
        random_state=73,
    )
    single_cv, single_sim = simulate_single_sup_lr_critical_values(
        np.eye(1), trim_ratio=.3, n_sim=5, grid_size=60, random_state=19)

    try:
        normalization_diff = float(np.max(np.abs(
            _normalize_factors(f*1e-8) - _normalize_factors(f))))
    except ValueError:
        normalization_diff = "FAIL: scale-dependent rank rejection"

    output = {
        "hac_independent_reference_max_abs_diff": hac_diffs,
        "joint_lr_statistic": float(joint['test_statistic']),
        "joint_cv": float(joint['critical_value']),
        "joint_breakpoints": joint['breakpoints'],
        "joint_reject_null": joint['reject_null'],
        "joint_all_critical_values": list(
            map(float, joint["critical_values"].values())
        ),
        "single_cv": single_cv.tolist(),
        "single_sim_first": float(single_sim[0]),
        "factor_scale_normalization_max_diff": normalization_diff,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
