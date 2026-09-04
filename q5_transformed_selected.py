#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5 当前选择方案：变换空间 / 静态烟幕点联合模型

当前高分辨率已验证候选：
    T1 = 21.675684249000 s
    T2 = 9.808648722000 s
    T3 = 6.544447336000 s
    J  = 38.028780307124 s

验证精度：
    dt=0.0025, nphi=720, nz=81

核心：
    S = E + (0,0,3 te)
    C(t) = S - (0,0,3 t)

同一时刻多烟幕联合判据：
    G_m(t) = max_j min_b dist(S_b, transformed LOS segment_(m,j)(t)) - 10
    G_m(t) <= 0 视为 M_m 完整遮蔽。

这是当前数值候选/下界，不宣称全局最优。
"""

from __future__ import annotations
import argparse
import math
import numpy as np
from scipy.optimize import brentq

G = 9.8
VC = 3.0
R = 10.0
T_CLOUD = 20.0
VM = 300.0

CYL_R = 7.0
CYL_CX = 0.0
CYL_CY = 200.0
CYL_Z0 = 0.0
CYL_Z1 = 10.0

M0S = np.array([
    [20000.0, 0.0, 2000.0],
    [19000.0, 600.0, 2100.0],
    [18000.0, -600.0, 1900.0],
], dtype=float)
VMS = -VM * M0S / np.linalg.norm(M0S, axis=1)[:, None]
T_HITS = np.linalg.norm(M0S, axis=1) / VM

UAVS = np.array([
    [17800.0, 0.0, 1800.0],
    [12000.0, 1400.0, 1400.0],
    [6000.0, -3000.0, 700.0],
    [11000.0, 2000.0, 1800.0],
    [13000.0, -2000.0, 1300.0],
], dtype=float)

STRATEGY = [{'name': 'FY1', 'V': 138.875501093989, 'theta': 3.137761983410153, 'bombs': [(0.001809742279331096, 0.00496017172415255, 0.006769914003483646, (17799.059832, 0.003601, 1800.020189)), (2.520869051253692, 4.889006175924048, 7.40987522717774, (16770.957415, 3.941942, 1705.107957)), (5.0203620193397915, 5.829606210464731, 10.849968229804523, (16293.216281, 5.77202, 1666.026793))]}, {'name': 'FY2', 'V': 140.0, 'theta': 5.026548245743669, 'bombs': [(5.163788049349122, 5.1520258113147825, 10.315813860663905, (12446.28665113, 26.47092097, 1300.88492878)), (6.879277991259239, 0.6095404953764874, 7.488818486635727, (12323.98410522, 402.87945305, 1420.64591134)), (9.975768895439698, 3.6145759069593697, 13.590344802399068, (12587.95065047, -409.52603762, 1376.75175537))]}, {'name': 'FY3', 'V': 139.92286419469065, 'theta': 1.785253310056994, 'bombs': [(16.318502071, 5.052234455, 21.370736525, (5363.623325, -78.245786, 639.039352)), (17.47113771, 5.09886481, 22.57000252, (5327.911645, 85.714893, 640.317738)), (19.42618945, 4.479162829, 23.905352278, (5288.147669, 268.280607, 673.407849))]}, {'name': 'FY4', 'V': 140.0, 'theta': 4.642575810304916, 'bombs': [(1.1019241826390704, 10.66389434130582, 11.76581852394489, (10885.09611847, 356.79793861, 1278.07610721)), (2.2295664916842703, 11.841957938743983, 14.071524430428253, (10862.57881057, 34.78543342, 1155.07793096))]}, {'name': 'FY5', 'V': 130.0, 'theta': 2.426007660272118, 'bombs': [(16.969374075174816, 6.740036726915889, 23.709410802090705, (10673.81646841, 22.12049374, 1148.53056651))]}]

def candidate_points(m, t, nphi, nz):
    M = M0S[m] + VMS[m] * t
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

    gens = []
    for ph in (alpha + delta, alpha - delta):
        gens.append(np.column_stack([
            np.full(nz, CYL_R*np.cos(ph)),
            np.full(nz, CYL_CY + CYL_R*np.sin(ph)),
            z
        ]))
    return M, np.vstack([top, bottom, *gens])

def segment_distances(M, P, C):
    B = P - M
    A = C - M
    lam = (B @ A) / np.sum(B*B, axis=1)
    lam = np.clip(lam, 0.0, 1.0)
    return np.linalg.norm(A[None, :] - lam[:, None]*B, axis=1)

def joint_violation(m, t, nphi, nz):
    M, P = candidate_points(m, t, nphi, nz)
    best = np.full(len(P), np.inf, dtype=float)

    for u in STRATEGY:
        for td, tau, te, S_tuple in u["bombs"]:
            if te <= t <= te + T_CLOUD:
                S = np.asarray(S_tuple, dtype=float)
                C = S - np.array([0.0, 0.0, VC*t])
                best = np.minimum(best, segment_distances(M, P, C))

    return float(np.max(best) - R)

def scan_intervals(m, dt, nphi, nz):
    """
    与搜索阶段最终复核相同的时间序列积分方式：
    在均匀时间网格上计算 G_m(t)，对相邻异号点线性插值端点。
    这样 --high 与当前记录使用同一套数值口径。
    """
    tmax = float(T_HITS[m])
    ts = np.arange(0.0, tmax + 1e-12, dt)
    vals = np.array([joint_violation(m, float(t), nphi, nz) for t in ts])

    good = vals <= 0.0
    out = []
    i = 0

    while i < len(ts):
        if not good[i]:
            i += 1
            continue

        start = float(ts[i])
        if i > 0 and np.isfinite(vals[i-1]) and vals[i-1] > 0.0 and vals[i] <= 0.0:
            f = vals[i-1] / (vals[i-1] - vals[i])
            start = float(ts[i-1] + f*(ts[i]-ts[i-1]))

        j = i
        while j + 1 < len(ts) and good[j+1]:
            j += 1

        end = float(ts[j])
        if j + 1 < len(ts) and np.isfinite(vals[j+1]) and vals[j] <= 0.0 and vals[j+1] > 0.0:
            f = (-vals[j]) / (vals[j+1] - vals[j])
            end = float(ts[j] + f*(ts[j+1]-ts[j]))

        out.append((start, end))
        i = j + 1

    return out


def bomb_geometry(i, V, theta, td, tau):
    F0 = UAVS[i]
    te = td + tau
    vF = np.array([V*np.cos(theta), V*np.sin(theta), 0.0])
    D = F0 + vF*td
    E = F0 + vF*te + np.array([0.0, 0.0, -0.5*G*tau*tau])
    S = E + np.array([0.0, 0.0, VC*te])
    return D, E, S

def print_data():
    for i, u in enumerate(STRATEGY):
        print(f"\n{u['name']}  V={u['V']:.12f} m/s  theta={math.degrees(u['theta'])%360:.9f} deg")
        for k, (td, tau, te, S0) in enumerate(sorted(u["bombs"], key=lambda x:x[0]), 1):
            D, E, S = bomb_geometry(i, u["V"], u["theta"], td, tau)
            print(f" bomb{k}: td={td:.9f} tau={tau:.9f} te={te:.9f}")
            print(f"   D = {D}")
            print(f"   E = {E}")
            print(f"   S = {S}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "high"], default="quick")
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()

    print_data()
    if args.data_only:
        return

    if args.mode == "quick":
        dt, nphi, nz = 0.01, 180, 31
    else:
        dt, nphi, nz = 0.0025, 720, 81

    print(f"\nverification: dt={dt}, nphi={nphi}, nz={nz}")
    total = 0.0
    for m in range(3):
        iv = scan_intervals(m, dt, nphi, nz)
        dur = sum(b-a for a,b in iv)
        total += dur
        print(f"M{m+1}: T={dur:.12f} s")
        for a,b in iv:
            print(f"  [{a:.12f}, {b:.12f}]  len={b-a:.12f}")
    print(f"J = {total:.12f} s")

if __name__ == "__main__":
    main()
