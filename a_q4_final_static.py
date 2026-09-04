#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 CUMCM A题 - 问题4（最终整理版）

FY1、FY2、FY3 各投放 1 枚烟幕弹，对 M1 实施干扰。

核心模型
--------
1. 导弹 M1 匀速直指假目标原点。
2. 无人机受领任务后等高度、匀速、直线飞行。
3. 烟幕弹继承无人机水平速度，投放后仅受重力。
4. 起爆后形成半径 R=10 m 的球形有效烟幕，持续 20 s，
   云团中心以 3 m/s 竖直下沉。
5. 真目标按“上圆周 + 下圆周 + 两条切母线”离散视轮廓。
6. 对每个轮廓点 P，判断烟幕中心到“导弹 M(t) -> P”的有限线段距离。
7. 多烟幕联合遮蔽：
       对每个轮廓点，至少有一个当前有效烟幕覆盖该视线段。
   即：
       max_j min_k d_{k,j}(t) <= R

本版额外使用“变换空间”：
    S = E + (0, 0, VC * te)
起爆后 S 为静态点；给定 S 后，自动反解“最早可实现”的无人机控制参数。
当前最优盆地中 FY2、FY3 均自动落到 V=140 m/s。

当前高精度已验证结果（6480×241 轮廓）：
    FY1: [0.924734148, 5.512785760]
    FY2/FY3 联合连续段:
         [18.917070336, 26.525285985]
    总有效遮蔽时间:
         12.196267261 s

依赖：
    numpy
    scipy
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


# ============================================================
# 1. 常量
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

VM = 300.0
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
vM = -VM * M0 / np.linalg.norm(M0)
T_HIT = float(np.linalg.norm(M0) / VM)

F_LIST = np.array(
    [
        [17800.0, 0.0, 1800.0],      # FY1
        [12000.0, 1400.0, 1400.0],   # FY2
        [6000.0, -3000.0, 700.0],    # FY3
    ],
    dtype=float,
)


# ============================================================
# 2. 当前最好策略
# ============================================================
#
# FY1 仍采用单弹最优。
#
# FY2/FY3 用变换空间静态点表示更自然：
#   S = E + (0,0,VC*te)
#
# 下面 BEST_STATIC 是由当前最好控制参数精确换算得到的。
# 程序会从 S 自动反解最早可实现控制参数，不直接硬编码 FY2/FY3 的
# (V, theta, td, tau)。
# ============================================================

FY1_X = np.array(
    [
        140.0,
        math.radians(5.164255048),
        0.772558664,
        0.152139506,
    ],
    dtype=float,
)

# 当前记录对应的静态烟幕点。
BEST_STATIC = np.array(
    [
        # FY2
        [9992.944224154, 55.439399232, 1062.757222288],
        # FY3
        [6583.911539698, 95.034257421, 735.559893557],
    ],
    dtype=float,
)


# ============================================================
# 3. 基本运动学
# ============================================================

@dataclass
class BombState:
    V: float
    theta: float
    td: float
    tau: float
    te: float
    vF: np.ndarray
    D: np.ndarray
    E: np.ndarray
    S: np.ndarray


def bomb_state(x: np.ndarray, F0: np.ndarray) -> BombState:
    V, theta, td, tau = map(float, x)
    te = td + tau

    vF = np.array(
        [V * np.cos(theta), V * np.sin(theta), 0.0],
        dtype=float,
    )

    D = F0 + vF * td
    E = F0 + vF * te + np.array(
        [0.0, 0.0, -0.5 * G * tau * tau],
        dtype=float,
    )

    S = E + np.array([0.0, 0.0, VC * te], dtype=float)

    return BombState(V, theta, td, tau, te, vF, D, E, S)


def valid_bomb(x: np.ndarray, F0: np.ndarray) -> bool:
    st = bomb_state(x, F0)
    return (
        70.0 <= st.V <= 140.0
        and st.td >= 0.0
        and st.tau >= 0.0
        and st.E[2] >= 0.0
        and st.te < T_HIT
    )


def earliest_control_from_static(S: np.ndarray, F0: np.ndarray) -> np.ndarray:
    """
    给定变换空间静态点 S，求“最早可实现”的 [V, theta, td, tau]。

    水平位移长度为 r，则
        70 <= r/te <= 140
    因此最快水平到达时间为 r/140。

    同时需满足竖直方向：
        S_z = z_F + VC*te - 1/2*g*tau^2
        0 <= tau <= te
        E_z = S_z - VC*te >= 0

    由于目标是最早实现该 S，所以最终最优常落在 V=140 的边界。
    """
    S = np.asarray(S, dtype=float)
    F0 = np.asarray(F0, dtype=float)

    dx = S[0] - F0[0]
    dy = S[1] - F0[1]
    r = float(np.hypot(dx, dy))

    if r <= 1e-12:
        raise ValueError("静态点水平位置与无人机初始位置重合，当前参数化不适用。")

    t_lo = max(
        r / 140.0,
        (S[2] - F0[2]) / VC,
        0.0,
    )
    t_hi = min(
        r / 70.0,
        S[2] / VC,
        T_HIT,
    )

    if t_lo > t_hi:
        raise ValueError("静态点不可达。")

    def tau_feasibility(te: float) -> float:
        fall = F0[2] + VC * te - S[2]

        if fall < 0.0:
            return -1e100

        # tau^2 = 2*fall/g，要求 tau <= te
        return te * te - 2.0 * fall / G

    if tau_feasibility(t_lo) < 0.0:
        if tau_feasibility(t_hi) < 0.0:
            raise ValueError("静态点竖直方向不可达。")

        t_lo = brentq(
            tau_feasibility,
            t_lo,
            t_hi,
            xtol=1e-12,
            rtol=1e-12,
        )

    te = float(t_lo)
    V = r / te
    theta = float(np.arctan2(dy, dx))

    fall = F0[2] + VC * te - S[2]
    tau = math.sqrt(max(0.0, 2.0 * fall / G))
    td = te - tau

    x = np.array([V, theta, td, tau], dtype=float)

    if not valid_bomb(x, F0):
        raise ValueError("反解得到的控制参数不满足物理约束。")

    return x


def build_best_strategy() -> np.ndarray:
    X = np.empty((3, 4), dtype=float)
    X[0] = FY1_X
    X[1] = earliest_control_from_static(BEST_STATIC[0], F_LIST[1])
    X[2] = earliest_control_from_static(BEST_STATIC[1], F_LIST[2])
    return X


# ============================================================
# 4. 圆柱视轮廓
# ============================================================

def candidate_points(t: float, nphi: int, nz: int):
    """
    当前时刻的圆柱视轮廓候选点：
      - 上圆周
      - 下圆周
      - 两条切母线
    """
    M = M0 + vM * t

    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    x = CYL_R * np.cos(phi)
    y = CYL_CY + CYL_R * np.sin(phi)

    top = np.column_stack(
        [x, y, np.full(nphi, CYL_Z1)]
    )
    bottom = np.column_stack(
        [x, y, np.full(nphi, CYL_Z0)]
    )

    dx = M[0] - CYL_CX
    dy = M[1] - CYL_CY

    rho = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)
    delta = np.arccos(np.clip(CYL_R / rho, -1.0, 1.0))

    z = np.linspace(CYL_Z0, CYL_Z1, nz)

    generators = []

    for ph in (alpha + delta, alpha - delta):
        generators.append(
            np.column_stack(
                [
                    np.full(nz, CYL_R * np.cos(ph)),
                    np.full(nz, CYL_CY + CYL_R * np.sin(ph)),
                    z,
                ]
            )
        )

    P = np.vstack([top, bottom, *generators])

    return M, P


# ============================================================
# 5. 有限线段距离判据
# ============================================================

def segment_distances(M: np.ndarray, P: np.ndarray, C: np.ndarray):
    """
    烟幕中心 C 到各有限视线段 [M,P_j] 的距离。

        lambda = (C-M)·(P-M) / ||P-M||^2
        lambda_c = clip(lambda, 0, 1)

    这是修正后的有限线段判据，不再把 0<=lambda<=1 当成硬约束。
    """
    B = P - M
    A = C - M

    B2 = np.sum(B * B, axis=1)
    lam = (B @ A) / B2
    lam_c = np.clip(lam, 0.0, 1.0)

    diff = A[None, :] - lam_c[:, None] * B

    return np.linalg.norm(diff, axis=1)


# ============================================================
# 6. 单烟幕 / 多烟幕违反量
# ============================================================

def smoke_center(st: BombState, t: float) -> np.ndarray:
    return st.E + np.array(
        [0.0, 0.0, -VC * (t - st.te)],
        dtype=float,
    )


def single_violation(
    x: np.ndarray,
    F0: np.ndarray,
    t: float,
    nphi: int,
    nz: int,
) -> float:
    """
    <= 0 : 单朵烟幕完整遮蔽整个视轮廓。
    """
    if not valid_bomb(x, F0):
        return np.inf

    st = bomb_state(x, F0)

    if not (st.te <= t <= min(T_HIT, st.te + T_CLOUD)):
        return np.inf

    M, P = candidate_points(t, nphi, nz)
    C = smoke_center(st, t)

    d = segment_distances(M, P, C)

    return float(np.max(d) - R)


def joint_violation(
    X: np.ndarray,
    t: float,
    nphi: int,
    nz: int,
) -> float:
    """
    真正的多烟幕联合判据：

        G(t) = max_j min_k d_{k,j}(t) - R

    <= 0 表示所有轮廓点都至少被一朵当前有效烟幕覆盖。
    """
    M, P = candidate_points(t, nphi, nz)

    best = np.full(len(P), np.inf, dtype=float)
    any_active = False

    for k in range(3):
        st = bomb_state(X[k], F_LIST[k])

        if st.te <= t <= min(T_HIT, st.te + T_CLOUD):
            any_active = True
            C = smoke_center(st, t)
            d = segment_distances(M, P, C)
            best = np.minimum(best, d)

    if not any_active:
        return np.inf

    return float(np.max(best) - R)


# ============================================================
# 7. 单烟幕完整遮蔽区间
# ============================================================

def roots_from_scan(fun, a: float, b: float, dt: float):
    xs = np.arange(a, b + 1e-12, dt)

    if xs[-1] < b:
        xs = np.append(xs, b)

    vs = np.array([fun(float(t)) for t in xs], dtype=float)

    roots = []

    for i in range(len(xs) - 1):
        v0 = vs[i]
        v1 = vs[i + 1]

        if (
            np.isfinite(v0)
            and np.isfinite(v1)
            and v0 * v1 < 0.0
        ):
            roots.append(
                float(
                    brentq(
                        fun,
                        xs[i],
                        xs[i + 1],
                        xtol=1e-12,
                        rtol=1e-12,
                    )
                )
            )

    return roots


def single_intervals(
    x: np.ndarray,
    F0: np.ndarray,
    nphi: int,
    nz: int,
    dt: float,
):
    st = bomb_state(x, F0)

    a = st.te
    b = min(T_HIT, st.te + T_CLOUD)

    fun = lambda t: single_violation(x, F0, t, nphi, nz)

    roots = roots_from_scan(fun, a, b, dt)
    cuts = [a] + roots + [b]

    out = []

    for u, v in zip(cuts[:-1], cuts[1:]):
        if fun(0.5 * (u + v)) <= 0.0:
            out.append((float(u), float(v)))

    return out


# ============================================================
# 8. 对当前最优拓扑的“防漏缝”验证
# ============================================================

@dataclass
class VerificationResult:
    early_interval: tuple[float, float]
    late_interval: tuple[float, float]
    total_duration: float
    fy2_interval: tuple[float, float]
    fy3_interval: tuple[float, float]
    first_bridge_margin: float
    overlap_max_violation: float
    overlap_worst_time: float


def verify_current_topology(
    X: np.ndarray,
    nphi: int,
    nz: int,
    root_dt: float,
    overlap_nphi: int,
    overlap_nz: int,
    overlap_dt: float,
) -> VerificationResult:
    """
    当前最优拓扑是：
        FY1 早期单独完整遮蔽
        FY2 后期先单独完整遮蔽
        FY3 起爆后与 FY2 空间互补
        FY3 最后单独完整遮蔽

    因此只需严格检查两个连接条件：

    1) FY2 单独完整遮蔽结束 b2 >= FY3 最早起爆 te3；
    2) 在 [te3, a3] 内：
           max_t G_23(t) <= 0
       其中 a3 为 FY3 单独完整遮蔽开始时刻。

    两者成立时，FY2/FY3 后半段就是一个严格连续区间 [a2,b3]。
    """
    i1 = single_intervals(
        X[0], F_LIST[0],
        nphi=nphi, nz=nz, dt=root_dt,
    )
    i2 = single_intervals(
        X[1], F_LIST[1],
        nphi=nphi, nz=nz, dt=root_dt,
    )
    i3 = single_intervals(
        X[2], F_LIST[2],
        nphi=nphi, nz=nz, dt=root_dt,
    )

    if len(i1) != 1 or len(i2) != 1 or len(i3) != 1:
        raise RuntimeError(
            f"当前拓扑假设不再成立：单烟幕区间数为 "
            f"{len(i1)}, {len(i2)}, {len(i3)}"
        )

    early = i1[0]
    a2, b2 = i2[0]
    a3, b3 = i3[0]

    te3 = bomb_state(X[2], F_LIST[2]).te

    first_bridge_margin = b2 - te3

    # 真正的 FY2/FY3 联合违反量，只在内部接力段扫描。
    X23 = X.copy()

    ts = np.arange(te3, a3 + 1e-12, overlap_dt)

    if ts[-1] < a3:
        ts = np.append(ts, a3)

    vals = np.array(
        [
            joint_violation(
                X23,
                float(t),
                nphi=overlap_nphi,
                nz=overlap_nz,
            )
            for t in ts
        ],
        dtype=float,
    )

    imax = int(np.argmax(vals))
    overlap_max_violation = float(vals[imax])
    overlap_worst_time = float(ts[imax])

    if first_bridge_margin < -1e-9:
        raise RuntimeError(
            f"FY2 在 FY3 起爆前已失效，存在时间裂缝："
            f"{-first_bridge_margin:.9f} s"
        )

    if overlap_max_violation > 0.0:
        raise RuntimeError(
            f"FY2/FY3 互补段存在隐藏裂缝："
            f"max G={overlap_max_violation:.9e} m, "
            f"t={overlap_worst_time:.9f} s"
        )

    late = (a2, b3)
    total = (early[1] - early[0]) + (late[1] - late[0])

    return VerificationResult(
        early_interval=early,
        late_interval=late,
        total_duration=float(total),
        fy2_interval=(a2, b2),
        fy3_interval=(a3, b3),
        first_bridge_margin=float(first_bridge_margin),
        overlap_max_violation=overlap_max_violation,
        overlap_worst_time=overlap_worst_time,
    )


# ============================================================
# 9. 输出
# ============================================================

def print_strategy(X: np.ndarray):
    print("===== Q4 strategy =====")

    for k in range(3):
        st = bomb_state(X[k], F_LIST[k])

        print(f"\nFY{k+1}")
        print(f"V           = {st.V:.12f} m/s")
        print(f"theta       = {np.degrees(st.theta) % 360.0:.9f} deg")
        print(f"td          = {st.td:.9f} s")
        print(f"tau         = {st.tau:.9f} s")
        print(f"te          = {st.te:.9f} s")
        print(f"D           = {st.D}")
        print(f"E           = {st.E}")
        print(f"S(static)   = {st.S}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["quick", "high", "ultra"],
        default="high",
        help="验证精度；ultra 最慢但用于最终复核。",
    )

    args = parser.parse_args()

    X = build_best_strategy()

    if args.mode == "quick":
        cfg = dict(
            nphi=360,
            nz=31,
            root_dt=0.01,
            overlap_nphi=720,
            overlap_nz=61,
            overlap_dt=0.001,
        )
    elif args.mode == "high":
        cfg = dict(
            nphi=2160,
            nz=121,
            root_dt=0.005,
            overlap_nphi=2160,
            overlap_nz=121,
            overlap_dt=0.0005,
        )
    else:
        cfg = dict(
            nphi=6480,
            nz=241,
            root_dt=0.0025,
            overlap_nphi=4320,
            overlap_nz=181,
            overlap_dt=0.0001,
        )

    print_strategy(X)

    print(f"\n===== verification: {args.mode} =====")
    result = verify_current_topology(X, **cfg)

    print(f"FY1 interval      = {result.early_interval}")
    print(f"FY2 alone         = {result.fy2_interval}")
    print(f"FY3 alone         = {result.fy3_interval}")
    print(f"FY2 -> FY3 margin = {result.first_bridge_margin:.9f} s")
    print(
        "overlap max G     = "
        f"{result.overlap_max_violation:.9e} m "
        f"at t={result.overlap_worst_time:.9f} s"
    )
    print(f"late joint interval = {result.late_interval}")
    print(f"\nTOTAL = {result.total_duration:.12f} s")


if __name__ == "__main__":
    main()
