#!/usr/bin/env python3
"""S-wave Coulomb bound state: Nyström quadrature + shooting.

Reconstructs the lecture Mathematica fragments (Gauss-Legendre on [0, 1],
p = tan(arc x), Jacobian weights) and solves Fig. 1's integral equation.

The physically equivalent radial ODE is also shot with RK4 / solve_ivp
as an independent check against E_n = - mu * alpha^2 / (2 n^2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

# Atomic units by default: hydrogen E_n = -1 / (2 n^2)
MU_DEFAULT = 1.0
ALPHA_DEFAULT = 1.0


def gauss_legendre_01(n: int):
    """n-point Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, 1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def lecture_five_point_grid():
    """Reproduce lecture Out[26], Out[31], Out[32]."""
    x, w = gauss_legendre_01(5)
    # Out[31] middle node is Tan[0.5 * arc] = 0.904988
    arc = 2.0 * np.arctan(0.904988)
    p = np.tan(arc * x)
    jac = arc * (1.0 + p ** 2)
    dq = np.abs(jac) * w
    return {
        "weights": w,
        "arc": arc,
        "qregion": p,
        "Deltaq": dq,
        "xregion": x,
    }


def momentum_grid(n: int, arc: float, p_scale: float = 1.0):
    x, w = gauss_legendre_01(n)
    p = p_scale * np.tan(arc * x)
    jac = p_scale * arc * (1.0 + np.tan(arc * x) ** 2)
    dp = np.abs(jac) * w
    pmax = p_scale * np.tan(arc)
    return x, w, p, dp, pmax


def kappa(pt, p, mex: float = 0.0):
    """Angular-integrated Coulomb/Yukawa kernel without the overall prefactor."""
    num = (pt + p) ** 2 + mex ** 2
    den = (pt - p) ** 2 + mex ** 2
    return (p / pt) * np.log(num / den)


def J_split(pt: float, pmax: float, mex: float = 0.0) -> float:
    """int_0^{pmax} kappa(pt, p) dp, split at the integrable log singularity."""

    def f(p):
        return kappa(pt, p, mex)

    if pt <= 0.0:
        val, _ = quad(f, 0.0, pmax, epsabs=1e-10, limit=200)
        return val
    if pt >= pmax:
        val, _ = quad(f, 0.0, pmax, epsabs=1e-10, limit=200)
        return val
    v1, _ = quad(f, 0.0, pt, epsabs=1e-10, limit=200)
    v2, _ = quad(f, pt, pmax, epsabs=1e-10, limit=200)
    return v1 + v2


def build_kernel(p, dp, pmax, alpha: float, mex: float = 0.0):
    """Nyström kernel with singularity subtraction.

    Fig. 1 prefactor: 4 pi^2 alpha / (2 pi)^3 = alpha / (2 pi).
    """
    pref = alpha / (2.0 * np.pi)
    n = len(p)
    Kreg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Kreg[i, j] = pref * dp[j] * kappa(p[i], p[j], mex)
    J = np.array([J_split(p[i], pmax, mex) for i in range(n)])
    K = Kreg.copy()
    row_sum = Kreg.sum(axis=1)
    np.fill_diagonal(K, pref * J - row_sum)
    return K


def nystrom_spectrum(n, arc, mu, alpha, p_scale=1.0, mex=0.0):
    x, w, p, dp, pmax = momentum_grid(n, arc, p_scale)
    K = build_kernel(p, dp, pmax, alpha, mex)
    T = np.diag(p ** 2 / (2.0 * mu))
    evals, evecs = np.linalg.eig(T - K)
    evals = np.real(evals)
    evecs = np.real(evecs)
    order = np.argsort(evals)
    return p, dp, pmax, K, evals[order], evecs[:, order]


def sigma_min(K, p, mu, energy):
    M = np.diag(p ** 2 / (2.0 * mu) - energy) - K
    return float(np.linalg.svd(M, compute_uv=False).min())


def shoot_integral_operator(K, p, mu, bracket):
    a, b = bracket

    def f(e):
        return sigma_min(K, p, mu, e)

    # sigma_min is non-negative; locate a dip by scanning, then
    # use a signed det (or a shifted SVD) for brentq.
    # Signed shooting function: least-squares residual with phase from det.
    def signed(e):
        M = np.diag(p ** 2 / (2.0 * mu) - e) - K
        sign, logdet = np.linalg.slogdet(M)
        return sign * np.exp(np.clip(logdet, -40.0, 40.0))

    fa, fb = signed(a), signed(b)
    if fa * fb > 0:
        # fall back: minimise sigma on a dense mesh inside the bracket
        es = np.linspace(a, b, 80)
        sigs = [f(e) for e in es]
        return float(es[int(np.argmin(sigs))])
    return float(brentq(signed, a, b, xtol=1e-12))


def radial_rhs(r, y, energy, mu, alpha):
    u, up = y
    upp = -2.0 * mu * (energy + alpha / r) * u
    return [up, upp]


def u_at_rmax(energy, mu=MU_DEFAULT, alpha=ALPHA_DEFAULT, rmax=40.0):
    eps = 1e-6
    try:
        sol = solve_ivp(
            radial_rhs,
            (eps, rmax),
            y0=(eps, 1.0),
            args=(energy, mu, alpha),
            rtol=1e-8,
            atol=1e-10,
            dense_output=False,
            events=lambda r, y, *rest: np.abs(y[0]) - 1e8,
        )
        return float(sol.y[0, -1])
    except Exception:
        return 1e8


def shoot_ode(bracket, mu=MU_DEFAULT, alpha=ALPHA_DEFAULT, rmax=40.0):
    def f(e):
        return u_at_rmax(e, mu, alpha, rmax)

    return float(brentq(f, bracket[0], bracket[1], xtol=1e-14))


def analytic_E(n, mu, alpha):
    return -mu * alpha ** 2 / (2.0 * n ** 2)


def analytic_u_1s(r):
    return 2.0 * r * np.exp(-r)


def analytic_phi_1s(p, gamma=1.0):
    return 1.0 / (p ** 2 + gamma ** 2) ** 2


def save_plots(out_dir: Path, p, phi, K, mu, alpha, e_shot, e_ode, scan_e, scan_s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(scan_e, scan_s, "o-", ms=4)
    ax.axvline(e_shot, color="C1", ls="--", label=f"shot E1 = {e_shot:.6f}")
    ax.axvline(analytic_E(1, mu, alpha), color="C2", ls=":", label="analytic -1/2")
    ax.set_xlabel("E")
    ax.set_ylabel(r"$\sigma_{\min}\,M(E)$")
    ax.set_title("Shooting function of the discretized integral operator")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "shooting_sigma_min.png", dpi=140)
    plt.close(fig)

    gamma = mu * alpha
    phi_an = analytic_phi_1s(p, gamma)
    # match overall scale and sign to the numerical ground state
    phi_n = phi / np.max(np.abs(phi))
    phi_an = phi_an / np.max(phi_an)
    if np.dot(phi_n, phi_an) < 0:
        phi_n = -phi_n
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(p, phi_n, "o", ms=4, label="Nyström $\\phi(p)$")
    p_fine = np.linspace(0, min(8.0, p.max()), 400)
    ax.plot(p_fine, analytic_phi_1s(p_fine, gamma) / analytic_phi_1s(0.0, gamma),
            label=r"analytic $1/(p^2+\gamma^2)^2$")
    ax.set_xlim(0, 8)
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\phi(p)$ (normalized)")
    ax.set_title("Momentum-space S-wave ground state")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "momentum_wavefunction.png", dpi=140)
    plt.close(fig)

    # coordinate-space u(r) from ODE at the shot energy
    eps, rmax = 1e-6, 20.0
    sol = solve_ivp(
        radial_rhs, (eps, rmax), y0=(eps, 1.0),
        args=(e_ode, mu, alpha), rtol=1e-8, atol=1e-10,
        dense_output=True, max_step=0.05,
    )
    r = np.linspace(eps, 12.0, 600)
    u_num = sol.sol(r)[0]
    u_an = analytic_u_1s(r)
    u_num = u_num / u_num[np.argmin(np.abs(r - 1.0))]
    u_an = u_an / analytic_u_1s(1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(r, u_num, label="ODE shooting $u(r)$")
    ax.plot(r, u_an, "--", label=r"analytic $2r e^{-r}$")
    ax.set_xlabel("r")
    ax.set_ylabel("u(r) (normalized at r=1)")
    ax.set_title("Coordinate-space radial wave function (ground state)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_wavefunction.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--arc", type=float, default=1.55)
    parser.add_argument("--mu", type=float, default=MU_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--p-scale", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== lecture 5-point grid (Out[26], Out[31], Out[32]) ===")
    lec = lecture_five_point_grid()
    print("weights:", np.array2string(lec["weights"], precision=6))
    print("arc    :", lec["arc"])
    print("qregion:", np.array2string(lec["qregion"], precision=6))
    print("Deltaq :", np.array2string(lec["Deltaq"], precision=6))
    w_ref = np.array([0.118463, 0.239314, 0.284444, 0.239314, 0.118463])
    p_ref = np.array([0.0691205, 0.353158, 0.904988, 2.1288, 5.87207])
    dq_ref = np.array([0.175107, 0.395971, 0.761169, 1.94754, 6.18349])
    assert np.allclose(lec["weights"], w_ref, atol=5e-7)
    assert np.allclose(lec["qregion"], p_ref, atol=5e-5)
    assert np.allclose(lec["Deltaq"], dq_ref, atol=5e-5)
    print("lecture grid matched.")

    print(f"\n=== Nyström kernel n={args.n}, arc={args.arc} ===")
    p, dp, pmax, K, evals, evecs = nystrom_spectrum(
        args.n, args.arc, args.mu, args.alpha, args.p_scale
    )
    print("pmax =", pmax)
    print("lowest 5 eigenvalues:", evals[:5])
    analytic = [analytic_E(n, args.mu, args.alpha) for n in range(1, 5)]
    print("analytic E_n        :", analytic)

    print("\n=== shooting the integral operator ===")
    e1 = shoot_integral_operator(K, p, args.mu, (-0.55, -0.40))
    e2 = shoot_integral_operator(K, p, args.mu, (-0.14, -0.10))
    print(f"shot E1 = {e1:.8f}   analytic = {analytic[0]:.8f}")
    print(f"shot E2 = {e2:.8f}   analytic = {analytic[1]:.8f}")

    scan_e = np.linspace(-0.70, -0.02, 35)
    scan_s = np.array([sigma_min(K, p, args.mu, e) for e in scan_e])

    print("\n=== coordinate-space ODE shooting ===")
    e1_ode = shoot_ode((-0.7, -0.3), args.mu, args.alpha, rmax=40.0)
    e2_ode = shoot_ode((-0.16, -0.08), args.mu, args.alpha, rmax=80.0)
    print(f"ODE  E1 = {e1_ode:.12f}   analytic = {analytic[0]:.12f}")
    print(f"ODE  E2 = {e2_ode:.12f}   analytic = {analytic[1]:.12f}")

    # write a small numeric report
    report = []
    report.append("# S-wave Coulomb bound-state results (atomic units)\n")
    report.append(f"mu = {args.mu}, alpha = {args.alpha}, nQuad = {args.n}, arc = {args.arc}\n\n")
    report.append("| n | analytic E | Nyström / integral shooting | ODE shooting |\n")
    report.append("|---|------------|-----------------------------|--------------|\n")
    shots = [e1, e2]
    odes = [e1_ode, e2_ode]
    for n in range(1, 5):
        ny = evals[n - 1] if n - 1 < len(evals) else float("nan")
        sh = shots[n - 1] if n - 1 < len(shots) else ny
        od = odes[n - 1] if n - 1 < len(odes) else float("nan")
        if n <= 2:
            report.append(
                f"| {n} | {analytic[n-1]:.8f} | {sh:.8f} | {od:.8f} |\n"
            )
        else:
            report.append(
                f"| {n} | {analytic[n-1]:.8f} | {ny:.8f} | — |\n"
            )
    report_path = out_dir / "energy_table.md"
    report_path.write_text("".join(report))
    print("\n" + "".join(report))

    save_plots(
        out_dir, p, evecs[:, 0], K, args.mu, args.alpha,
        e1, e1_ode, scan_e, scan_s,
    )
    # energy bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"n={n}" for n in range(1, 5)]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.2, analytic, 0.4, label="analytic $-\\mu\\alpha^2/(2n^2)$")
    ax.bar(x + 0.2, evals[:4], 0.4, label="Nyström (subtraction)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("E")
    ax.set_title("Bound-state energies")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_comparison.png", dpi=140)
    plt.close(fig)
    print(f"figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
