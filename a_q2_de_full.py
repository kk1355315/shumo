import math
import time
import numpy as np
from scipy.optimize import brentq, differential_evolution

# =========================
# Constants
# =========================
G = 9.8
VC = 3.0
R_SMOKE = 10.0
T_SMOKE = 20.0

RCYL = 7.0
CX, CY = 0.0, 200.0
Z0, Z1 = 0.0, 10.0

M0 = np.array([20000.0, 0.0, 2000.0])
VM = 300.0
F0 = np.array([17800.0, 0.0, 1800.0])

vM = -VM * M0 / np.linalg.norm(M0)
MISSILE_HIT_TIME = float(np.linalg.norm(M0) / VM)
TAU_MAX = math.sqrt(2.0 * F0[2] / G)

# x = [VF, theta, td, tau]

def build_state(x):
    VF, theta, td, tau = map(float, x)

    vF = np.array([
        VF * np.cos(theta),
        VF * np.sin(theta),
        0.0
    ])

    te = td + tau
    Me = M0 + vM * te

    E = (
        F0
        + vF * te
        + np.array([0.0, 0.0, -0.5 * G * tau**2])
    )

    return te, Me, E


def violation_grid(x, nphi=72, nz=9, ns=101):
    VF, theta, td, tau = map(float, x)
    te, Me, E = build_state(x)

    if not (70.0 <= VF <= 140.0):
        return None, None
    if td < 0.0 or tau < 0.0:
        return None, None
    if E[2] < 0.0:
        return None, None
    if te >= MISSILE_HIT_TIME:
        return None, None

    smax = min(T_SMOKE, MISSILE_HIT_TIME - te)
    if smax <= 0:
        return None, None

    s = np.linspace(0.0, smax, ns)

    M = Me[None, :] + s[:, None] * vM[None, :]
    C = E[None, :] + s[:, None] * np.array([0.0, 0.0, -VC])[None, :]
    A = C - M

    # -------------------------
    # upper/lower rims
    # -------------------------
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    circle_x = RCYL * np.cos(phi)
    circle_y = CY + RCYL * np.sin(phi)

    top = np.column_stack([
        circle_x,
        circle_y,
        np.full(nphi, Z1)
    ])

    bottom = np.column_stack([
        circle_x,
        circle_y,
        np.full(nphi, Z0)
    ])

    P = np.vstack([top, bottom])

    B = P[None, :, :] - M[:, None, :]
    B2 = np.sum(B * B, axis=2)
    lam = np.sum(B * A[:, None, :], axis=2) / B2
    lam_c = np.clip(lam, 0.0, 1.0)
    dseg = np.linalg.norm(
        A[:, None, :] - lam_c[:, :, None] * B,
        axis=2
    )

    max_dseg = dseg.max(axis=1)

    # -------------------------
    # two tangent generators
    # -------------------------
    dx = M[:, 0] - CX
    dy = M[:, 1] - CY

    Dh = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)
    delta = np.arccos(np.clip(RCYL / Dh, -1.0, 1.0))

    phis = np.stack([
        alpha + delta,
        alpha - delta
    ], axis=1)

    z = np.linspace(Z0, Z1, nz)

    Pgx = RCYL * np.cos(phis)[:, :, None] + np.zeros((1, 1, nz))
    Pgy = CY + RCYL * np.sin(phis)[:, :, None] + np.zeros((1, 1, nz))
    Pgz = np.broadcast_to(z[None, None, :], (ns, 2, nz))

    Pg = np.stack([Pgx, Pgy, Pgz], axis=3)

    Bg = Pg - M[:, None, None, :]
    B2g = np.sum(Bg * Bg, axis=3)
    lamg = np.sum(Bg * A[:, None, None, :], axis=3) / B2g
    lamg_c = np.clip(lamg, 0.0, 1.0)
    dseg_g = np.linalg.norm(
        A[:, None, None, :] - lamg_c[:, :, :, None] * Bg,
        axis=3
    )

    max_dseg = np.maximum(max_dseg, dseg_g.max(axis=(1, 2)))
    violation = max_dseg - R_SMOKE

    return s, violation


def violation_scalar(x, s, nphi=72, nz=9):
    te, Me, E = build_state(x)

    M = Me + vM * s
    C = E + np.array([0.0, 0.0, -VC]) * s
    A = C - M

    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    circle_x = RCYL * np.cos(phi)
    circle_y = CY + RCYL * np.sin(phi)

    top = np.column_stack([
        circle_x,
        circle_y,
        np.full(nphi, Z1)
    ])

    bottom = np.column_stack([
        circle_x,
        circle_y,
        np.full(nphi, Z0)
    ])

    dx = M[0] - CX
    dy = M[1] - CY
    Dh = np.hypot(dx, dy)

    alpha = np.arctan2(dy, dx)
    delta = np.arccos(np.clip(RCYL / Dh, -1.0, 1.0))

    z = np.linspace(Z0, Z1, nz)

    generators = []
    for ph in (alpha + delta, alpha - delta):
        generators.append(
            np.column_stack([
                np.full(nz, RCYL * np.cos(ph)),
                np.full(nz, CY + RCYL * np.sin(ph)),
                z
            ])
        )

    P = np.vstack([top, bottom, *generators])

    B = P - M
    B2 = np.sum(B * B, axis=1)
    lam = (B @ A) / B2
    lam_c = np.clip(lam, 0.0, 1.0)
    dseg = np.linalg.norm(A[None, :] - lam_c[:, None] * B, axis=1)

    return float(dseg.max() - R_SMOKE)


def duration_and_min_violation(x, nphi=72, nz=9, ns=101):
    s, v = violation_grid(x, nphi=nphi, nz=nz, ns=ns)

    if s is None:
        return 0.0, 1e3

    min_v = float(v.min())
    roots = []

    for i in range(len(s) - 1):
        if v[i] == 0.0:
            roots.append(float(s[i]))

        if v[i] * v[i + 1] < 0.0:
            root = brentq(
                lambda ss: violation_scalar(
                    x,
                    ss,
                    nphi=nphi,
                    nz=nz
                ),
                float(s[i]),
                float(s[i + 1]),
                xtol=1e-10,
                rtol=1e-10
            )
            roots.append(float(root))

    cuts = sorted(
        set(
            [float(s[0]), float(s[-1])]
            + [round(r, 10) for r in roots]
        )
    )

    duration = 0.0

    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + b)

        if violation_scalar(
            x,
            mid,
            nphi=nphi,
            nz=nz
        ) <= 0.0:
            duration += b - a

    return float(duration), min_v


def make_objective(nphi, nz, ns):
    def objective(x):
        duration, min_v = duration_and_min_violation(
            x,
            nphi=nphi,
            nz=nz,
            ns=ns
        )

        # If a valid shielding interval already exists, maximize its length.
        if duration > 0.0:
            return -duration

        # Otherwise use minimum constraint violation to guide DE toward feasibility.
        return min_v

    return objective


BOUNDS = [
    (70.0, 140.0),                   # VF
    (-np.pi, np.pi),                 # theta
    (0.0, MISSILE_HIT_TIME),         # td
    (0.0, TAU_MAX)                   # tau
]


def global_de(seed=7):
    objective = make_objective(
        nphi=72,
        nz=9,
        ns=101
    )

    return differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        popsize=8,
        maxiter=60,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-7,
        atol=1e-7,
    )


def fine_de(x0):
    VF0, th0, td0, tau0 = x0

    fine_bounds = [
        (max(70.0, VF0 - 5.0), 140.0),
        (max(-np.pi, th0 - 0.12), min(np.pi, th0 + 0.12)),
        (max(0.0, td0 - 1.5), min(MISSILE_HIT_TIME, td0 + 1.5)),
        (max(0.0, tau0 - 0.8), min(TAU_MAX, tau0 + 0.8)),
    ]

    objective = make_objective(
        nphi=180,
        nz=21,
        ns=181
    )

    return differential_evolution(
        objective,
        bounds=fine_bounds,
        seed=123,
        popsize=8,
        maxiter=50,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-9,
        atol=1e-9,
    )


if __name__ == "__main__":
    t0 = time.perf_counter()

    # Run several global seeds to reduce the chance of getting stuck in a poorer basin.
    results = [global_de(seed) for seed in (7, 21, 42)]

    best_global = min(
        results,
        key=lambda r: -duration_and_min_violation(
            r.x,
            nphi=180,
            nz=21,
            ns=181
        )[0]
    )

    fine = fine_de(best_global.x)

    # Optional last narrow refinement around the best point found above.
    VF0, th0, td0, tau0 = fine.x
    ultra_bounds = [
        (max(70.0, VF0 - 0.5), min(140.0, VF0 + 0.5)),
        (max(-np.pi, th0 - 0.03), min(np.pi, th0 + 0.03)),
        (max(0.0, td0 - 0.5), min(MISSILE_HIT_TIME, td0 + 0.5)),
        (max(0.0, tau0 - 0.3), min(TAU_MAX, tau0 + 0.3)),
    ]

    ultra_objective = make_objective(
        nphi=180,
        nz=21,
        ns=181
    )

    ultra = differential_evolution(
        ultra_objective,
        bounds=ultra_bounds,
        seed=2025,
        popsize=8,
        maxiter=60,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-10,
        atol=1e-10,
    )

    x = ultra.x

    duration, _ = duration_and_min_violation(
        x,
        nphi=720,
        nz=61,
        ns=301
    )

    te, Me, E = build_state(x)
    VF, theta, td, tau = x

    vF = np.array([
        VF * np.cos(theta),
        VF * np.sin(theta),
        0.0
    ])

    D = F0 + vF * td

    print("best parameters")
    print("VF =", VF)
    print("theta(rad) =", theta)
    print("theta(deg) =", np.degrees(theta))
    print("td =", td)
    print("tau =", tau)
    print("te =", te)

    print("\ndrop point D =", D)
    print("burst point E =", E)

    print("\nshield duration =", duration)
    print("runtime =", time.perf_counter() - t0)
