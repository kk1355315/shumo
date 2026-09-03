import math
import time
import numpy as np
from scipy.optimize import differential_evolution, brentq, minimize, minimize_scalar

# ============================================================
# A题 Q3：FY1 投放 3 枚烟幕弹干扰 M1
# 判据：圆柱上圆周 + 下圆周 + 两条切母线
# 多云团联合遮蔽：对每个轮廓点，至少有一个有效烟幕球
# 与“导弹-该点”视线段相交；所有轮廓点均满足时计为有效遮蔽。
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
F0 = np.array([17800.0, 0.0, 1800.0])

vM = -VM * M0 / np.linalg.norm(M0)
T_HIT = float(np.linalg.norm(M0) / VM)
TAU_MAX = math.sqrt(2.0 * F0[2] / G)

# x = [VF, theta, td1, g12, g23, tau1, tau2, tau3]
# td2 = td1 + 1 + g12
# td3 = td2 + 1 + g23
# 由此自动满足同一无人机相邻投弹间隔 >= 1 s


def unpack(x):
    VF, theta, td1, g12, g23, tau1, tau2, tau3 = map(float, x)
    td2 = td1 + 1.0 + g12
    td3 = td2 + 1.0 + g23
    td = np.array([td1, td2, td3])
    tau = np.array([tau1, tau2, tau3])
    te = td + tau
    return VF, theta, td, tau, te


def build_bombs(x):
    VF, theta, td, tau, te = unpack(x)
    vF = np.array([VF*np.cos(theta), VF*np.sin(theta), 0.0])

    # 起爆点。烟幕弹投放后继承无人机水平速度并受重力作用。
    E = (
        F0[None, :]
        + te[:, None] * vF[None, :]
        + np.column_stack([
            np.zeros(3),
            np.zeros(3),
            -0.5 * G * tau**2
        ])
    )
    D = F0[None, :] + td[:, None] * vF[None, :]
    return VF, theta, td, tau, te, vF, D, E


def valid_strategy(x):
    VF, theta, td, tau, te, vF, D, E = build_bombs(x)
    if not (70.0 <= VF <= 140.0):
        return False
    if np.any(td < 0.0) or np.any(tau < 0.0):
        return False
    if np.any(tau > TAU_MAX):
        return False
    if td[1] - td[0] < 1.0 - 1e-12:
        return False
    if td[2] - td[1] < 1.0 - 1e-12:
        return False
    if np.any(te >= T_HIT):
        return False
    if np.any(E[:, 2] < 0.0):
        return False
    return True


# ============================================================
# 预计算粗网格：仅用于 DE 搜索
# ============================================================

def precompute_geometry(dt=0.18, nphi=24, nz=5):
    times = np.arange(0.0, T_HIT + 1e-12, dt)
    if times[-1] < T_HIT:
        times = np.append(times, T_HIT)

    M = M0[None, :] + times[:, None] * vM[None, :]
    nt = len(times)

    # 上下圆周
    phi = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)
    x = CYL_R * np.cos(phi)
    y = CYL_CY + CYL_R * np.sin(phi)

    top = np.column_stack([x, y, np.full(nphi, CYL_Z1)])
    bottom = np.column_stack([x, y, np.full(nphi, CYL_Z0)])
    rims = np.vstack([top, bottom])
    rims = np.broadcast_to(rims[None, :, :], (nt, 2*nphi, 3))

    # 两条切母线
    dx = M[:, 0] - CYL_CX
    dy = M[:, 1] - CYL_CY
    rho = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)
    delta = np.arccos(np.clip(CYL_R / rho, -1.0, 1.0))

    phis = np.stack([alpha + delta, alpha - delta], axis=1)
    z = np.linspace(CYL_Z0, CYL_Z1, nz)

    gx = CYL_R*np.cos(phis)[:, :, None] + np.zeros((1, 1, nz))
    gy = CYL_CY + CYL_R*np.sin(phis)[:, :, None] + np.zeros((1, 1, nz))
    gz = np.broadcast_to(z[None, None, :], (nt, 2, nz))

    gens = np.stack([gx, gy, gz], axis=3).reshape(nt, 2*nz, 3)

    P = np.concatenate([rims, gens], axis=1)
    B = P - M[:, None, :]
    B2 = np.sum(B*B, axis=2)

    return times, M, B, B2


T_GRID, M_GRID, B_GRID, B2_GRID = precompute_geometry()


def duration_from_samples(times, violation):
    total = 0.0
    for i in range(len(times)-1):
        t0, t1 = times[i], times[i+1]
        v0, v1 = violation[i], violation[i+1]

        if v0 <= 0.0 and v1 <= 0.0:
            total += t1 - t0
        elif np.isfinite(v0) and np.isfinite(v1) and v0*v1 < 0.0:
            tr = t0 + (t1-t0) * (-v0) / (v1-v0)
            total += (tr-t0) if v0 <= 0.0 else (t1-tr)

    return float(total)


def coarse_duration(x):
    if not valid_strategy(x):
        return 0.0, 1e3

    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    d_all = []
    for k in range(3):
        C = E[k][None, :] + (T_GRID-te[k])[:, None] * np.array([0.0, 0.0, -VC])
        A = C - M_GRID

        # 点到线段距离：
        # lambda_c = clip((A·B)/|B|^2, 0, 1)
        # d_seg = |A - lambda_c B|
        lam = np.einsum("tpi,ti->tp", B_GRID, A) / B2_GRID
        lam_c = np.clip(lam, 0.0, 1.0)
        dseg = np.linalg.norm(
            A[:, None, :] - lam_c[:, :, None] * B_GRID,
            axis=2
        )

        active = (T_GRID >= te[k]) & (T_GRID <= te[k] + T_CLOUD)
        dseg[~active, :] = np.inf
        d_all.append(dseg)

    # 每个目标点取 3 枚烟幕中距离最近的一枚；
    # 再对全部轮廓点取最坏情况。
    d_best = np.minimum.reduce(d_all)
    violation = np.max(d_best, axis=1) - R

    duration = duration_from_samples(T_GRID, violation)

    finite = np.isfinite(violation)
    min_violation = float(np.min(violation[finite])) if np.any(finite) else 1e3
    return duration, min_violation


def objective(x):
    duration, min_violation = coarse_duration(x)
    if duration > 0.0:
        return -duration
    return min_violation


# ============================================================
# 高精度动态轮廓判据
# ============================================================

def candidate_points(t, nphi=720, nz=61):
    M = M0 + vM*t

    phi = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)
    x = CYL_R*np.cos(phi)
    y = CYL_CY + CYL_R*np.sin(phi)

    top = np.column_stack([x, y, np.full(nphi, CYL_Z1)])
    bottom = np.column_stack([x, y, np.full(nphi, CYL_Z0)])

    dx = M[0] - CYL_CX
    dy = M[1] - CYL_CY
    rho = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)
    delta = np.arccos(np.clip(CYL_R/rho, -1.0, 1.0))

    z = np.linspace(CYL_Z0, CYL_Z1, nz)
    generators = []
    for ph in (alpha+delta, alpha-delta):
        generators.append(
            np.column_stack([
                np.full(nz, CYL_R*np.cos(ph)),
                np.full(nz, CYL_CY + CYL_R*np.sin(ph)),
                z
            ])
        )

    return M, np.vstack([top, bottom, *generators])


def exact_violation(x, t, nphi=720, nz=61):
    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    M, P = candidate_points(t, nphi=nphi, nz=nz)
    B = P - M
    B2 = np.sum(B*B, axis=1)

    best = np.full(len(P), np.inf)

    for k in range(3):
        if not (te[k] <= t <= te[k] + T_CLOUD):
            continue

        C = E[k] + np.array([0.0, 0.0, -VC])*(t-te[k])
        A = C - M

        lam = (B @ A) / B2
        lam_c = np.clip(lam, 0.0, 1.0)
        dseg = np.linalg.norm(A[None, :] - lam_c[:, None]*B, axis=1)

        best = np.minimum(best, dseg)

    return float(np.max(best) - R)


def exact_intervals(x, nphi=720, nz=61, dt=0.025):
    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    # 起爆/失效时刻加入事件点，避免跨越 active set 变化直接求根。
    events = [0.0, T_HIT]
    for k in range(3):
        events += [te[k], min(te[k] + T_CLOUD, T_HIT)]
    events = sorted(set(max(0.0, min(float(e), T_HIT)) for e in events))

    roots = []

    for a, b in zip(events[:-1], events[1:]):
        if b-a <= 1e-12:
            continue

        n = max(2, int(np.ceil((b-a)/dt)) + 1)
        xs = np.linspace(a, b, n)
        vs = np.array([exact_violation(x, tt, nphi, nz) for tt in xs])

        for i in range(len(xs)-1):
            if not np.isfinite(vs[i]) or not np.isfinite(vs[i+1]):
                continue

            if vs[i]*vs[i+1] < 0.0:
                roots.append(
                    brentq(
                        lambda tt: exact_violation(x, tt, nphi, nz),
                        xs[i], xs[i+1],
                        xtol=1e-11,
                        rtol=1e-11
                    )
                )

    cuts = sorted(set(events + [round(float(r), 11) for r in roots]))

    intervals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b-a <= 1e-12:
            continue
        if exact_violation(x, 0.5*(a+b), nphi, nz) <= 0.0:
            intervals.append([a, b])

    merged = []
    for a, b in intervals:
        if not merged or a-merged[-1][1] > 1e-8:
            merged.append([a, b])
        else:
            merged[-1][1] = b

    return [(float(a), float(b)) for a, b in merged]


# ============================================================
# DE
# ============================================================

BOUNDS = [
    (70.0, 140.0),        # VF
    (-np.pi, np.pi),      # theta
    (0.0, 8.0),           # td1
    (0.0, 12.0),          # g12
    (0.0, 12.0),          # g23
    (0.0, 12.0),          # tau1
    (0.0, 12.0),          # tau2
    (0.0, 12.0),          # tau3
]

# 物理启发初值：尽早投放，满足 1 s 最小间隔。
# 第三枚设置为 t=2 s 立即起爆；DE 仍可在完整边界内调整全部 8 个变量。
X0 = np.array([
    140.0,
    np.deg2rad(5.0),
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])


def solve():
    t0 = time.perf_counter()

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=2026,
        popsize=6,
        maxiter=45,
        polish=False,
        x0=X0,
        updating="immediate",
        workers=1,
        tol=1e-8,
        atol=1e-8,
    )

    # 粗网格排序可能造成很小的排名误差，因此将
    # DE 最优解与物理启发初值都做高密度复核，取更优者。
    candidates = [result.x, X0]
    verified = []

    for x in candidates:
        intervals = exact_intervals(x)
        duration = sum(b-a for a, b in intervals)
        verified.append((duration, x.copy(), intervals))

    duration, x_best, intervals = max(verified, key=lambda z: z[0])

    VF, theta, td, tau, te, vF, D, E = build_bombs(x_best)

    runtime = time.perf_counter() - t0

    print("===== Q3 =====")
    print(f"DE nfev = {result.nfev}, nit = {result.nit}")
    print(f"runtime = {runtime:.3f} s")
    print(f"VF = {VF:.9f} m/s")
    print(f"theta = {theta:.9f} rad = {np.degrees(theta):.9f} deg")
    print()

    for k in range(3):
        print(
            f"bomb {k+1}: "
            f"td={td[k]:.9f}, tau={tau[k]:.9f}, te={te[k]:.9f}"
        )
        print(f"  drop  = {D[k]}")
        print(f"  burst = {E[k]}")

    print()
    print("coverage intervals =", intervals)
    print(f"TOTAL = {duration:.12f} s")

    return {
        "x": x_best,
        "duration": duration,
        "intervals": intervals,
        "VF": VF,
        "theta": theta,
        "td": td,
        "tau": tau,
        "te": te,
        "D": D,
        "E": E,
        "runtime": runtime,
        "nfev": result.nfev,
        "nit": result.nit,
    }

# ============================================================
# Q3 endpoint refinement (final solver)
# ============================================================

# ============================================================
# Q3 endpoint optimizer
#
# Core idea:
#   1. For each smoke bomb, solve its complete-shield interval [a_i, b_i]
#      using Brent root finding, not a time-grid objective.
#   2. In the good basin, bombs 2 and 3 are already fully shielding at
#      their burst instants. Therefore enforce:
#          te2 = b1
#          te3 = b2
#      which eliminates g12 and g23 from the outer optimization.
#   3. Outer variables:
#          theta, td1, tau1, tau2, tau3
#      with VF fixed at 140 m/s after repeated searches hit the upper bound.
# ============================================================

VF_FIXED = 140.0


def single_violation(x, k, t, nphi=90, nz=11):
    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    if not (te[k] <= t <= te[k] + T_CLOUD):
        return np.inf

    M, P = candidate_points(t, nphi=nphi, nz=nz)
    B = P - M
    B2 = np.sum(B * B, axis=1)

    C = E[k] + np.array([0.0, 0.0, -VC]) * (t - te[k])
    A = C - M

    lam = (B @ A) / B2
    lam_c = np.clip(lam, 0.0, 1.0)

    dseg = np.linalg.norm(
        A[None, :] - lam_c[:, None] * B,
        axis=1
    )

    return float(dseg.max() - R)


def single_interval(x, k, nphi=90, nz=11, dt_scan=0.08):
    if not valid_strategy(x):
        return []

    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    a = max(0.0, float(te[k]))
    b = min(T_HIT, float(te[k] + T_CLOUD))

    if b <= a:
        return []

    xs = np.arange(a, b + 1e-12, dt_scan)
    if xs[-1] < b:
        xs = np.append(xs, b)

    vs = np.array([
        single_violation(x, k, t, nphi, nz)
        for t in xs
    ])

    roots = []
    for i in range(len(xs) - 1):
        v0, v1 = vs[i], vs[i + 1]

        if (
            np.isfinite(v0)
            and np.isfinite(v1)
            and v0 * v1 < 0.0
        ):
            roots.append(
                brentq(
                    lambda tt: single_violation(
                        x, k, tt, nphi, nz
                    ),
                    xs[i],
                    xs[i + 1],
                    xtol=1e-9,
                    rtol=1e-9,
                )
            )

    cuts = [a] + roots + [b]
    intervals = []

    for u, v in zip(cuts[:-1], cuts[1:]):
        if single_violation(
            x, k, 0.5 * (u + v), nphi, nz
        ) <= 0.0:
            intervals.append((float(u), float(v)))

    return intervals


def single_ab(x, k, nphi=90, nz=11, dt_scan=0.08):
    intervals = single_interval(
        x, k, nphi, nz, dt_scan
    )

    if not intervals:
        return None

    return intervals[0][0], intervals[-1][1]


def build_chained(
    theta,
    td1,
    tau1,
    tau2,
    tau3,
    nphi=90,
    nz=11,
    dt_scan=0.08,
):
    # temporary gaps; overwritten below
    x = np.array([
        VF_FIXED,
        theta,
        td1,
        0.0,
        0.0,
        tau1,
        tau2,
        tau3,
    ])

    # bomb 1
    ab1 = single_ab(
        x, 0, nphi, nz, dt_scan
    )
    if ab1 is None:
        return None

    a1, b1 = ab1

    # bomb 2: force burst exactly at bomb 1's end
    td2 = b1 - tau2
    g12 = td2 - td1 - 1.0

    if g12 < 0.0:
        return None

    x[3] = g12

    ab2 = single_ab(
        x, 1, nphi, nz, dt_scan
    )
    if ab2 is None:
        return None

    a2, b2 = ab2

    # bomb 3: force burst exactly at bomb 2's end
    td3 = b2 - tau3
    g23 = td3 - td2 - 1.0

    if g23 < 0.0:
        return None

    x[4] = g23

    if not valid_strategy(x):
        return None

    ab3 = single_ab(
        x, 2, nphi, nz, dt_scan
    )
    if ab3 is None:
        return None

    a3, b3 = ab3

    return x, (a1, b1, a2, b2, a3, b3), b3 - a1


def objective(y):
    theta, td1, tau1, tau2, tau3 = y

    result = build_chained(
        theta,
        td1,
        tau1,
        tau2,
        tau3,
        nphi=90,
        nz=11,
        dt_scan=0.08,
    )

    if result is None:
        return 1e3

    return -result[2]


# ============================================================
# Q3 final v2: nested endpoint optimization
#
# Repeated searches show:
#   VF -> 140 m/s
#   td1 -> 0 s
#   optimal handoff is essentially seamless:
#       te2 = b1
#       te3 = b2
#
# For a fixed theta and tau1:
#   - solve bomb 1 interval [a1,b1]
#   - choose tau2 to maximize b2 with te2=b1
#   - choose tau3 to maximize b3 with te3=b2
# Outer optimization only needs theta and tau1.
# ============================================================

VF_FINAL = 140.0
TD1_FINAL = 0.0


def nested_solution(
    theta,
    tau1,
    nphi=90,
    nz=11,
    dt_scan=0.08,
):
    # Bomb 1
    x = np.array([
        VF_FINAL, theta, TD1_FINAL,
        0.0, 0.0,
        tau1, 5.34, 6.05
    ])

    ab1 = single_ab(
        x, 0, nphi, nz, dt_scan
    )
    if ab1 is None:
        return None

    a1, b1 = ab1

    # Bomb 2: burst exactly when bomb 1 complete shielding ends.
    def objective_tau2(tau2):
        td2 = b1 - tau2
        g12 = td2 - TD1_FINAL - 1.0

        if g12 < 0.0:
            return 1e3

        xt = x.copy()
        xt[3] = g12
        xt[6] = tau2

        ab2 = single_ab(
            xt, 1, nphi, nz, dt_scan
        )
        if ab2 is None:
            return 1e3

        if ab2[0] > b1 + 1e-5:
            return 1e3

        return -ab2[1]

    r2 = minimize_scalar(
        objective_tau2,
        bounds=(5.0, 5.65),
        method="bounded",
        options={
            "xatol": 2e-6,
            "maxiter": 40,
        },
    )

    if r2.fun >= 1e2:
        return None

    tau2 = float(r2.x)
    b2 = -float(r2.fun)

    td2 = b1 - tau2
    g12 = td2 - TD1_FINAL - 1.0

    # Bomb 3: burst exactly when bomb 2 complete shielding ends.
    def objective_tau3(tau3):
        td3 = b2 - tau3
        g23 = td3 - td2 - 1.0

        if g23 < 0.0:
            return 1e3

        xt = np.array([
            VF_FINAL, theta, TD1_FINAL,
            g12, g23,
            tau1, tau2, tau3
        ])

        if not valid_strategy(xt):
            return 1e3

        ab3 = single_ab(
            xt, 2, nphi, nz, dt_scan
        )
        if ab3 is None:
            return 1e3

        if ab3[0] > b2 + 1e-5:
            return 1e3

        return -ab3[1]

    r3 = minimize_scalar(
        objective_tau3,
        bounds=(5.75, 6.35),
        method="bounded",
        options={
            "xatol": 2e-6,
            "maxiter": 40,
        },
    )

    if r3.fun >= 1e2:
        return None

    tau3 = float(r3.x)
    b3 = -float(r3.fun)

    td3 = b2 - tau3
    g23 = td3 - td2 - 1.0

    x_final = np.array([
        VF_FINAL, theta, TD1_FINAL,
        g12, g23,
        tau1, tau2, tau3
    ])

    return x_final, b3 - a1


def final_objective(y):
    theta, tau1 = y

    result = nested_solution(
        theta,
        tau1,
        nphi=90,
        nz=11,
        dt_scan=0.08,
    )

    if result is None:
        return 1e3

    return -result[1]


def solve_v2():
    # Optimize only the two remaining outer variables.
    y0 = np.array([
        np.deg2rad(179.6487814),
        3.6034290,
    ])

    bounds = [
        (
            np.deg2rad(179.62),
            np.deg2rad(179.68),
        ),
        (3.55, 3.66),
    ]

    result = minimize(
        final_objective,
        y0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 30,
            "xtol": 5e-7,
            "ftol": 1e-11,
            "disp": False,
        },
    )

    # Rebuild with a denser endpoint calculation.
    dense = nested_solution(
        result.x[0],
        result.x[1],
        nphi=180,
        nz=21,
        dt_scan=0.05,
    )

    x = dense[0]

    # Final full-joint high-density verification.
    intervals = exact_intervals(
        x,
        nphi=4320,
        nz=181,
        dt=0.00375,
    )

    duration = sum(
        b - a
        for a, b in intervals
    )

    VF, theta, td, tau, te, vF, D, E = build_bombs(x)

    print("===== Q3 final v2 =====")
    print("VF =", VF)
    print("theta_deg =", np.degrees(theta))
    print("td =", td)
    print("tau =", tau)
    print("te =", te)
    print("D =")
    print(D)
    print("E =")
    print(E)
    print("intervals =", intervals)
    print("duration =", duration)

    return x, intervals, duration


if __name__ == "__main__":
    solve_v2()
