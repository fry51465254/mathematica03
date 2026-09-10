#!/usr/bin/env python3
"""S-wave bound state following the two lecture Mathematica snippets.

未命名-2: Gauss nodes on [0, 1], p = tan(x π/2), Jacobian weights Δq.
未命名-1: eqleft == λ + 4π²α Σ Δq (p/pt) log(...) φ(p), then
          λ(Eb) = LinearSolve(...)[0] and shoot λ(Eb) = 0.

Fig. 1's kernel (not the inverted log / missing (2π)³ from the live slide)
is used so that a bound state exists. mex regulates the p = pt diagonal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

MU_DEFAULT = 1.0
ALPHA_DEFAULT = 1.0 / 137.0
MEX_DEFAULT = 1.0e-3


def gauss_legendre_01(n: int):
    """Stand-in for NIntegrate`GaussRuleData[n, prec] on [0, 1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def lecture_grid(n: int):
    """未命名-2: p = tan(x π/2), Δq = |∂p/∂x| w.

    The slide wrote (jacobi /. xt -> weights); jacobi depends on x, so
    the substitution must be xt -> absc.
    """
    absc, weights = gauss_legendre_01(n)
    absc = np.minimum(np.asarray(absc, dtype=float), 1.0 - 1e-12)
    weights = np.asarray(weights, dtype=float)
    arc = 0.5 * np.pi
    qregion = np.tan(arc * absc)
    jacobi = arc * (1.0 + qregion ** 2)
    deltaq = np.abs(jacobi * weights)
    return absc, weights, qregion, deltaq


def kernel_matrix(qregion, deltaq, alpha, mex, include_twopi3=True, invert_log=False):
    """Nyström matrix for the Sum in eqright."""
    pref = 4.0 * np.pi ** 2 * alpha
    if include_twopi3:
        pref /= (2.0 * np.pi) ** 3
    pt = qregion[:, None]
    pj = qregion[None, :]
    num = (pt + pj) ** 2 + mex ** 2
    den = (pt - pj) ** 2 + mex ** 2
    logk = np.log(den / num) if invert_log else np.log(num / den)
    return pref * deltaq[None, :] * (pj / pt) * logk


def linear_system(Eb, qregion, K, mu, i0):
    """Discrete form of eqleft == λ + K φ, plus φ[i0] == 1.

    Unknowns x = [λ, φ_1, ..., φ_n]. Returns the same first component
    that LinearSolve[a, b][[1]] extracts in the lecture snippet.
    """
    n = len(qregion)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    kinetic = qregion ** 2 / (2.0 * mu) - Eb
    A[:n, 0] = -1.0
    A[:n, 1:] = np.diag(kinetic) - K
    A[n, 1 + i0] = 1.0
    rhs[n] = 1.0
    return A, rhs


def lambda_of_Eb(Eb, qregion, K, mu, i0):
    A, rhs = linear_system(Eb, qregion, K, mu, i0)
    sol = np.linalg.solve(A, rhs)
    return float(sol[0]), sol[1:]


def yukawa_u_inf(energy, mu, alpha, mex, rmax):
    eps = 1e-8

    def rhs(r, y):
        u, up = y
        return [up, -2.0 * mu * (energy + alpha * np.exp(-mex * r) / r) * u]

    sol = solve_ivp(
        rhs, (eps, rmax), y0=(eps, 1.0), rtol=1e-7, atol=1e-9,
        events=lambda r, y: np.abs(y[0]) - 1e8,
    )
    return float(sol.y[0, -1])


def analytic_coulomb(n, mu, alpha):
    return -mu * alpha ** 2 / (2.0 * n ** 2)


def save_plots(out_dir: Path, qregion, phi, scan_e, scan_lam, e_shot, e_ode,
               mu, alpha, mex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    gamma = mu * alpha
    a0 = 1.0 / gamma

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(scan_e, scan_lam, "o-", ms=4)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axvline(e_shot, color="C1", ls="--", label=f"λ-shot E1 = {e_shot:.3e}")
    ax.axvline(e_ode, color="C2", ls=":", label=f"ODE E1 = {e_ode:.3e}")
    ax.set_xlabel(r"$E_b$")
    ax.set_ylabel(r"$\lambda(E_b)$")
    ax.set_title("Lecture shooting function  λ(Eb) = LinearSolve[[1]]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_shooting.png", dpi=140)
    plt.close(fig)

    phi_n = phi / np.max(np.abs(phi))
    phi_an = 1.0 / (qregion ** 2 + gamma ** 2) ** 2
    phi_an = phi_an / np.max(phi_an)
    if np.dot(phi_n, phi_an) < 0:
        phi_n = -phi_n
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(qregion, phi_n, "o", ms=3, label=r"λ-shot $\phi(p)$")
    p_fine = np.linspace(0.0, 8.0 * gamma, 400)
    ax.plot(
        p_fine,
        (1.0 / (p_fine ** 2 + gamma ** 2) ** 2)
        / (1.0 / (gamma ** 4)),
        label=r"Coulomb 1s  $1/(p^2+\gamma^2)^2$",
    )
    ax.set_xlim(0.0, 8.0 * gamma)
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\phi(p)$ (normalized)")
    ax.set_title("Momentum-space ground state")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "momentum_wavefunction.png", dpi=140)
    plt.close(fig)

    rmax_plot = 6.0 * a0
    sol = solve_ivp(
        lambda r, y: [
            y[1],
            -2.0 * mu * (e_ode + alpha * np.exp(-mex * r) / r) * y[0],
        ],
        (1e-8, rmax_plot),
        y0=(1e-8, 1.0),
        rtol=1e-7,
        atol=1e-9,
        dense_output=True,
        max_step=0.2 * a0,
    )
    r = np.linspace(1e-8, rmax_plot, 800)
    u = sol.sol(r)[0]
    u = u / u[np.argmin(np.abs(r - a0))]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(r / a0, u, label="Yukawa ODE shooting")
    ax.set_xlabel(r"$r/a_0$")
    ax.set_ylabel("u(r) (normalized at $a_0$)")
    ax.set_title("Coordinate-space radial wave function")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_wavefunction.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = ["Coulomb analytic", "λ(Eb) shooting", "Yukawa ODE"]
    vals = [analytic_coulomb(1, mu, alpha), e_shot, e_ode]
    ax.bar(labels, vals, color=["C0", "C1", "C2"])
    ax.set_ylabel("E")
    ax.set_title(rf"Ground state  ($\alpha=1/137$, $m_x={mex:g}$)")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_comparison.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=160, help="nq in the lecture snippet")
    parser.add_argument("--mu", type=float, default=MU_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--mex", type=float, default=MEX_DEFAULT)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    absc, weights, qregion, deltaq = lecture_grid(args.n)
    print("=== 未命名-2 grid  p = tan(x π/2) ===")
    print(f"nq = {args.n}  pmin = {qregion[0]:.6e}  pmax = {qregion[-1]:.4f}")
    print(f"mex = {args.mex}  alpha = 1/137  mu = {args.mu}")

    K = kernel_matrix(qregion, deltaq, args.alpha, args.mex)
    i0 = int(np.argmin(np.abs(qregion - args.mu * args.alpha)))
    print(f"normalize φ at i0 = {i0}, p = {qregion[i0]:.6f}  (target μ α = {args.mu * args.alpha:.6f})")

    Ecoul = analytic_coulomb(1, args.mu, args.alpha)
    print(f"\nCoulomb analytic E1 = {Ecoul:.8e}")

    def lam(E):
        return lambda_of_Eb(E, qregion, K, args.mu, i0)[0]

    scan_e = np.linspace(2.0 * Ecoul, -1e-7, 36)
    scan_lam = np.array([lam(E) for E in scan_e])
    print("=== λ(Eb) scan ===")
    for E, L in list(zip(scan_e, scan_lam))[::5]:
        print(f"  Eb = {E:+.6e}   λ = {L:+.6e}")

    e_shot = brentq(lam, 1.3 * Ecoul, 0.4 * Ecoul)
    lam_shot, phi = lambda_of_Eb(e_shot, qregion, K, args.mu, i0)
    print(f"\nλ-shot E1 = {e_shot:.8e}   λ = {lam_shot:.3e}")

    a0 = 1.0 / (args.mu * args.alpha)
    rmax = 25.0 * a0

    def uinf(E):
        return yukawa_u_inf(E, args.mu, args.alpha, args.mex, rmax)

    e_ode = brentq(uinf, 1.3 * Ecoul, 0.4 * Ecoul)
    print(f"ODE-shot E1  = {e_ode:.8e}")

    report = (
        "# S-wave Yukawa / Coulomb bound state (lecture λ-shooting)\n\n"
        f"mu = {args.mu}, alpha = 1/137, mex = {args.mex}, nq = {args.n}\n\n"
        "| method | E1 |\n|---|---|\n"
        f"| Coulomb analytic $-\\mu\\alpha^2/2$ | {Ecoul:.8e} |\n"
        f"| λ(Eb) LinearSolve shooting | {e_shot:.8e} |\n"
        f"| Yukawa radial ODE shooting | {e_ode:.8e} |\n"
    )
    (out_dir / "energy_table.md").write_text(report)
    print("\n" + report)

    save_plots(
        out_dir, qregion, phi, scan_e, scan_lam, e_shot, e_ode,
        args.mu, args.alpha, args.mex,
    )
    print(f"figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
