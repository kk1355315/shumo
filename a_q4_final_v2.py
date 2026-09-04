import math
import numpy as np
from scipy.optimize import brentq

# ============================================================
# 2025 CUMCM A - Q4
# FY1 / FY2 / FY3 each release one smoke bomb against M1.
#
# Model:
#   - missile flies straight toward the dummy target
#   - UAV flies level, straight, constant speed after assignment
#   - bomb inherits UAV horizontal velocity, then falls under gravity
#   - after burst, smoke center descends vertically at 3 m/s
#   - effective smoke sphere radius = 10 m, lifetime = 20 s
#   - complete shielding criterion:
#       upper rim + lower rim + two tangent generators
#       for every sampled silhouette point, at least one active smoke sphere
#       intersects the missile-to-target-point line segment
#
# Current numerical strategy:
#   first solve the three one-bomb subproblems independently.
#   Their optimal complete-shield intervals are far apart in time, so the
#   full three-cloud joint verification equals the sum of those intervals.
# ============================================================

G = 9.8
VC = 3.0
R = 10.0
T_CLOUD = 20.0

CYL_R = 7.0
CYL_CX = 0.0
CYL_CY = 200.0
CYL_Z0 = 0.0
CYL_Z1 = 10.0

M0 = np.array([20000.0, 0.0, 2000.0])
VM = 300.0
vM = -VM * M0 / np.linalg.norm(M0)
T_HIT = float(np.linalg.norm(M0) / VM)

F_LIST = np.array([
    [17800.0,    0.0, 1800.0],   # FY1
    [12000.0, 1400.0, 1400.0],   # FY2
    [ 6000.0,-3000.0,  700.0],   # FY3
], dtype=float)

# [VF, theta(rad), td, tau]
# These are the polished numerical solutions found for the three one-bomb
# subproblems.  The final full-joint criterion is re-evaluated below.
BEST_X = np.array([
    [140.000000000000,  0.090133254000000,  0.772558664000000, 0.152139506000000],
    [140.000000000000, -0.901509752929274,  8.278336091856996, 4.169689294827052],
    [140.000000000000,  1.278316521216224, 23.053362184087064, 0.000000000000000],
], dtype=float)


def bomb_state(x, F0):
    VF, theta, td, tau = map(float, x)

    vF = np.array([
        VF*np.cos(theta),
        VF*np.sin(theta),
        0.0,
    ])

    te = td + tau

    D = F0 + vF*td

    E = (
        F0
        + vF*te
        + np.array([0.0, 0.0, -0.5*G*tau*tau])
    )

    return vF, te, D, E


def valid_bomb(x, F0):
    VF, theta, td, tau = map(float, x)
    vF, te, D, E = bomb_state(x, F0)

    return (
        70.0 <= VF <= 140.0
        and td >= 0.0
        and tau >= 0.0
        and E[2] >= 0.0
        and te < T_HIT
    )


def candidate_points(t, nphi=2160, nz=121):
    """
    Cylinder silhouette candidates:
      1) upper rim
      2) lower rim
      3) two tangent generators from current missile horizontal position
    """
    M = M0 + vM*t

    phi = np.linspace(
        0.0,
        2.0*np.pi,
        nphi,
        endpoint=False,
    )

    x = CYL_R*np.cos(phi)
    y = CYL_CY + CYL_R*np.sin(phi)

    top = np.column_stack([
        x,
        y,
        np.full(nphi, CYL_Z1),
    ])

    bottom = np.column_stack([
        x,
        y,
        np.full(nphi, CYL_Z0),
    ])

    dx = M[0] - CYL_CX
    dy = M[1] - CYL_CY

    rho = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)

    delta = np.arccos(
        np.clip(
            CYL_R/rho,
            -1.0,
            1.0,
        )
    )

    z = np.linspace(
        CYL_Z0,
        CYL_Z1,
        nz,
    )

    generators = []

    for ph in (
        alpha + delta,
        alpha - delta,
    ):
        generators.append(
            np.column_stack([
                np.full(nz, CYL_R*np.cos(ph)),
                np.full(nz, CYL_CY + CYL_R*np.sin(ph)),
                z,
            ])
        )

    P = np.vstack([
        top,
        bottom,
        *generators,
    ])

    return M, P


def segment_distances(M, P, C):
    """
    Distance from smoke-center C to each line segment M--P.
    This is the corrected segment criterion:
        lambda_c = clip(lambda, 0, 1).
    """
    B = P - M
    A = C - M

    B2 = np.sum(B*B, axis=1)

    lam = (B @ A) / B2
    lam_c = np.clip(lam, 0.0, 1.0)

    Qdiff = A[None, :] - lam_c[:, None]*B

    return np.linalg.norm(
        Qdiff,
        axis=1,
    )


def single_violation(x, F0, t, nphi=2160, nz=121):
    """
    max silhouette segment-distance minus smoke radius.
    <= 0 means this one smoke cloud completely shields the target.
    """
    if not valid_bomb(x, F0):
        return np.inf

    vF, te, D, E = bomb_state(x, F0)

    if not (
        te <= t <= min(T_HIT, te + T_CLOUD)
    ):
        return np.inf

    M, P = candidate_points(
        t,
        nphi=nphi,
        nz=nz,
    )

    C = (
        E
        + np.array([0.0, 0.0, -VC])*(t - te)
    )

    d = segment_distances(M, P, C)

    return float(
        np.max(d) - R
    )


def single_intervals(
    x,
    F0,
    nphi=2160,
    nz=121,
    dt=0.0075,
):
    """
    Find complete-shield intervals for one smoke cloud by:
      coarse time scan -> Brent roots -> interval midpoint test.
    """
    if not valid_bomb(x, F0):
        return []

    vF, te, D, E = bomb_state(x, F0)

    a = te
    b = min(T_HIT, te + T_CLOUD)

    xs = np.arange(
        a,
        b + 1e-12,
        dt,
    )

    if xs[-1] < b:
        xs = np.append(xs, b)

    vs = np.array([
        single_violation(
            x,
            F0,
            t,
            nphi,
            nz,
        )
        for t in xs
    ])

    roots = []

    for i in range(len(xs) - 1):
        v0 = vs[i]
        v1 = vs[i + 1]

        if (
            np.isfinite(v0)
            and np.isfinite(v1)
            and v0*v1 < 0.0
        ):
            root = brentq(
                lambda tt: single_violation(
                    x,
                    F0,
                    tt,
                    nphi,
                    nz,
                ),
                xs[i],
                xs[i + 1],
                xtol=1e-11,
                rtol=1e-11,
            )

            roots.append(float(root))

    cuts = [a] + roots + [b]
    out = []

    for u, v in zip(
        cuts[:-1],
        cuts[1:],
    ):
        mid = 0.5*(u + v)

        if single_violation(
            x,
            F0,
            mid,
            nphi,
            nz,
        ) <= 0.0:
            out.append(
                (float(u), float(v))
            )

    return out


def joint_violation(
    X,
    t,
    nphi=2160,
    nz=121,
):
    """
    True Q4 multi-cloud criterion:
        for every silhouette point,
        at least one ACTIVE smoke cloud must cover its line segment.
    """
    M, P = candidate_points(
        t,
        nphi=nphi,
        nz=nz,
    )

    best = np.full(
        len(P),
        np.inf,
    )

    any_active = False

    for k in range(3):
        x = X[k]
        F0 = F_LIST[k]

        if not valid_bomb(x, F0):
            return np.inf

        vF, te, D, E = bomb_state(x, F0)

        if (
            te <= t <= min(T_HIT, te + T_CLOUD)
        ):
            any_active = True

            C = (
                E
                + np.array([0.0, 0.0, -VC])*(t - te)
            )

            d = segment_distances(
                M,
                P,
                C,
            )

            best = np.minimum(
                best,
                d,
            )

    if not any_active:
        return np.inf

    return float(
        np.max(best) - R
    )


def joint_intervals(
    X,
    nphi=2160,
    nz=121,
    dt=0.0075,
):
    """
    High-density final verification of all three smoke clouds together.
    """
    events = [
        0.0,
        T_HIT,
    ]

    for k in range(3):
        vF, te, D, E = bomb_state(
            X[k],
            F_LIST[k],
        )

        events += [
            max(0.0, te),
            min(T_HIT, te + T_CLOUD),
        ]

    events = sorted(
        set(events)
    )

    roots = []

    for a, b in zip(
        events[:-1],
        events[1:],
    ):
        if b <= a:
            continue

        mid = 0.5*(a + b)

        if not np.isfinite(
            joint_violation(
                X,
                mid,
                nphi,
                nz,
            )
        ):
            continue

        xs = np.arange(
            a,
            b + 1e-12,
            dt,
        )

        if xs[-1] < b:
            xs = np.append(
                xs,
                b,
            )

        vs = np.array([
            joint_violation(
                X,
                t,
                nphi,
                nz,
            )
            for t in xs
        ])

        for i in range(
            len(xs) - 1
        ):
            v0 = vs[i]
            v1 = vs[i + 1]

            if (
                np.isfinite(v0)
                and np.isfinite(v1)
                and v0*v1 < 0.0
            ):
                roots.append(
                    brentq(
                        lambda tt: joint_violation(
                            X,
                            tt,
                            nphi,
                            nz,
                        ),
                        xs[i],
                        xs[i + 1],
                        xtol=1e-11,
                        rtol=1e-11,
                    )
                )

    cuts = sorted(
        set(
            events
            + [float(r) for r in roots]
        )
    )

    out = []

    for a, b in zip(
        cuts[:-1],
        cuts[1:],
    ):
        if b <= a:
            continue

        mid = 0.5*(a + b)

        v = joint_violation(
            X,
            mid,
            nphi,
            nz,
        )

        if (
            np.isfinite(v)
            and v <= 0.0
        ):
            if (
                out
                and a - out[-1][1] < 1e-8
            ):
                out[-1] = (
                    out[-1][0],
                    float(b),
                )
            else:
                out.append(
                    (float(a), float(b))
                )

    return out


def solve():
    X = BEST_X.copy()

    rows = []

    print("===== Q4 strategy =====")

    for k in range(3):
        x = X[k]
        F0 = F_LIST[k]

        vF, te, D, E = bomb_state(
            x,
            F0,
        )

        intervals = single_intervals(
            x,
            F0,
        )

        duration = sum(
            b - a
            for a, b in intervals
        )

        theta_deg = (
            np.degrees(x[1]) % 360.0
        )

        rows.append({
            "uav": f"FY{k+1}",
            "VF": x[0],
            "theta_deg": theta_deg,
            "td": x[2],
            "tau": x[3],
            "te": te,
            "D": D,
            "E": E,
            "intervals": intervals,
            "duration": duration,
        })

        print(f"\nFY{k+1}")
        print("VF =", x[0])
        print("theta_deg =", theta_deg)
        print("td =", x[2])
        print("tau =", x[3])
        print("te =", te)
        print("D =", D)
        print("E =", E)
        print("individual intervals =", intervals)
        print("individual duration =", duration)

    intervals = joint_intervals(
        X,
    )

    duration = sum(
        b - a
        for a, b in intervals
    )

    print("\n===== joint verification =====")
    print("joint intervals =", intervals)
    print("joint duration =", duration)

    return X, rows, intervals, duration


if __name__ == "__main__":
    solve()
