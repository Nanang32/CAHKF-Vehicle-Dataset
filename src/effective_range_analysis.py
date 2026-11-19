"""
effective_range_analysis.py
=============================
Kuantifikasi "effective range" metode estimasi percepatan dari lebar bbox:
pada jarak berapa akurasi mulai anjlok, dan rekomendasi eksplisit jarak
operasi yang andal.

DUA JALUR ANALISIS:
  A. ANALITIK -- turunkan sigma_D(D) dari rumus propagasi error
     (sg_acceleration_estimation.py), tunjukkan suku noise piksel tumbuh
     ~D^4 (jauh lebih cepat dari suku kalibrasi ~D^2), cari titik potong
     D_cross = sqrt(A/B) sbg kandidat definisi effective range yg berdasar
     fisika, bukan angka arbitrer.
  B. EMPIRIS -- sweep simulasi Kalman (spt postsession_analysis.py) di
     berbagai D0, ukur bias & RMSE relatif estimasi percepatan thd nilai
     benar, pada beberapa ambang toleransi (25%/50%/100%) -- validasi
     silang thd hasil analitik A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# A. Analitik: sigma_D(D) dan titik potong
# =============================================================================

def sigma_D_analytical(D, f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px):
    """sigma_D(D) lengkap (bukan cuma suku dominan) -- utk plot kurva penuh."""
    w = f_px * W_real_m / D
    rel_var = (sigma_f_px / f_px) ** 2 + (sigma_W_m / W_real_m) ** 2 \
              + (sigma_w_px / w) ** 2
    return D * np.sqrt(rel_var)


def crossover_distance(f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px):
    """
    D_cross: jarak di mana kontribusi suku kalibrasi (~D^2 di sigma_D^2)
    SAMA BESAR dgn kontribusi suku noise piksel (~D^4 di sigma_D^2).
    Di atas D_cross, noise piksel mendominasi & error tumbuh jauh lebih
    cepat (mendekati D^2 di sigma_D, bukan D^1).
    """
    A = (sigma_f_px / f_px) ** 2 + (sigma_W_m / W_real_m) ** 2
    B = (sigma_w_px / (f_px * W_real_m)) ** 2
    return float(np.sqrt(A / B))


def run_analytical_analysis(f_px=800.0, W_real_m=1.8,
                             sigma_f_px=2.0, sigma_W_m=0.05, sigma_w_px=2.5,
                             label="controlled (sigma_W_m=0.05)"):
    D_grid = np.linspace(3, 150, 300)
    sigma_D = sigma_D_analytical(D_grid, f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px)
    rel_error_pct = 100 * sigma_D / D_grid

    D_cross = crossover_distance(f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px)

    print(f"\n[ANALITIK -- {label}]")
    print(f"  f_px={f_px}, W_real_m={W_real_m}, sigma_f_px={sigma_f_px}, "
          f"sigma_W_m={sigma_W_m}, sigma_w_px={sigma_w_px}")
    print(f"  D_cross (noise piksel mulai dominan) = {D_cross:.1f} m")
    for target_pct in [5, 10, 20, 50]:
        idx = np.argmin(np.abs(rel_error_pct - target_pct))
        print(f"  sigma_D/D = {target_pct}% tercapai pada D ~= {D_grid[idx]:.1f} m")

    return D_grid, sigma_D, rel_error_pct, D_cross


# =============================================================================
# B. Empiris: Kalman filter + smoother (identik dgn postsession_analysis.py)
# =============================================================================

def _F(dt):
    return np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]])


def _Q(dt, q):
    return q * np.array([
        [dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
        [dt ** 3 / 2, dt ** 2, dt],
        [dt ** 2 / 2, dt, 1.0],
    ])


_H = np.array([[1.0, 0.0, 0.0]])


def kf_filter(t, z, q, r):
    n = len(t)
    r_arr = np.full(n, r) if np.isscalar(r) else np.asarray(r)
    x = np.array([z[0], 0.0, 0.0])
    P = np.diag([r_arr[0] * 10, 100.0, 100.0])
    x_filt, P_filt = np.zeros((n, 3)), np.zeros((n, 3, 3))
    innovations = np.full(n, np.nan)
    x_filt[0], P_filt[0] = x, P
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        F, Q = _F(dt), _Q(dt, q)
        x_pred, P_pred = F @ x, F @ P @ F.T + Q
        S = (_H @ P_pred @ _H.T)[0, 0] + r_arr[k]
        innovation = z[k] - (_H @ x_pred)[0]
        innovations[k] = innovation
        K = (P_pred @ _H.T).flatten() / S
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, _H @ P_pred)
        x_filt[k], P_filt[k] = x, P
    return dict(x_filt=x_filt, P_filt=P_filt, innovations=innovations)


def rts_smoother(t, filt, q):
    n = len(t)
    x_filt, P_filt = filt["x_filt"], filt["P_filt"]
    x_smooth = x_filt.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F, Q = _F(dt), _Q(dt, q)
        P_pred_next = F @ P_filt[k] @ F.T + Q
        C = P_filt[k] @ F.T @ np.linalg.pinv(P_pred_next)
        x_pred_next = F @ x_filt[k]
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred_next)
    return x_smooth


def innovation_rmse(filt):
    return float(np.sqrt(np.nanmean(filt["innovations"] ** 2)))


def tune_q(t, z, r, q_grid=None):
    if q_grid is None:
        q_grid = np.logspace(-3, 3, 20)
    best_q, best_rmse, best_filt = None, np.inf, None
    for q in q_grid:
        filt = kf_filter(t, z, q, r)
        rmse = innovation_rmse(filt)
        if rmse < best_rmse:
            best_q, best_rmse, best_filt = q, rmse, filt
    return best_q, best_filt


def estimate_R(D_meas, w_px, f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px):
    rel_var = (sigma_f_px / f_px) ** 2 + (sigma_W_m / W_real_m) ** 2 \
              + (sigma_w_px / w_px) ** 2
    return (D_meas * np.sqrt(rel_var)) ** 2


def safe_trial_duration(D0, v0, true_a, floor_m=4.0, default_max_s=4.0):
    """
    Hitung durasi trial maksimum yg AMAN (D(t) tidak menembus floor_m)
    utk kombinasi D0/v0/true_a tertentu -- supaya jarak dekat (D0 kecil)
    tetap bisa diuji tanpa lintasan sintetis menabrak batas deteksi.
    floor_m diberi margin (4.0, bukan 3.0 persis) sbg buffer aman.
    """
    a_coef, b_coef, c_coef = 0.5 * true_a, v0, (D0 - floor_m)
    if abs(a_coef) < 1e-9:
        if abs(b_coef) < 1e-9:
            return default_max_s
        t_root = -c_coef / b_coef
        return default_max_s if t_root <= 0 else min(default_max_s, 0.9 * t_root)
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0:
        return default_max_s  # lintasan tidak pernah menyentuh floor
    sqrt_disc = np.sqrt(disc)
    roots = [(-b_coef + sqrt_disc) / (2 * a_coef), (-b_coef - sqrt_disc) / (2 * a_coef)]
    positive_roots = [r for r in roots if r > 0]
    if not positive_roots:
        return default_max_s
    t_root = min(positive_roots)
    return min(default_max_s, 0.9 * t_root)


def run_single_trial(D0, true_a, f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px,
                      fps=15, duration_s=None, v0=-0.5, seed=0, min_frames=14):
    """Satu trial singkat (manuver terkontrol) dimulai pada jarak D0.
    duration_s=None -> dihitung adaptif via safe_trial_duration() supaya
    valid jg utk D0 kecil (dekat batas deteksi)."""
    if duration_s is None:
        duration_s = safe_trial_duration(D0, v0, true_a, floor_m=4.0, default_max_s=4.0)
    n = int(fps * duration_s)
    if n < min_frames:  # trial terlalu pendek utk window SG/Kalman yg berarti
        return np.nan

    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    D_true = D0 + v0 * t + 0.5 * true_a * t ** 2
    if D_true.min() < 3.0:
        return np.nan
    w_true = (f_px * W_real_m) / D_true
    w_noisy = w_true + rng.normal(0, sigma_w_px, n)
    D_meas = (f_px * W_real_m) / w_noisy

    R_t = estimate_R(D_meas, w_noisy, f_px, W_real_m, sigma_f_px, sigma_W_m, sigma_w_px)
    best_q, filt = tune_q(t, D_meas, R_t)
    x_smooth = rts_smoother(t, filt, best_q)

    marker_mid = duration_s / 2.0
    idx = int(np.argmin(np.abs(t - marker_mid)))
    return float(x_smooth[idx, 2])


def run_empirical_sweep(D0_grid, true_a=-1.5, n_trials=12,
                         f_px=800.0, W_real_m=1.8,
                         sigma_f_px=2.0, sigma_W_m=0.05, sigma_w_px=2.5,
                         label="controlled"):
    print(f"\n[EMPIRIS -- {label}] sweep D0={list(D0_grid)}, true_a={true_a}, "
          f"n_trials={n_trials} (durasi trial ADAPTIF utk jarak dekat)")
    rows = []
    for D0 in D0_grid:
        estimates = []
        dur_used = None
        for trial in range(n_trials):
            dur_used = safe_trial_duration(D0, v0=-0.5, true_a=true_a)
            est = run_single_trial(D0, true_a, f_px, W_real_m,
                                    sigma_f_px, sigma_W_m, sigma_w_px,
                                    duration_s=dur_used,
                                    seed=trial * 97 + int(D0 * 10))
            if not np.isnan(est):
                estimates.append(est)
        estimates = np.array(estimates)
        if len(estimates) == 0:
            print(f"  D0={D0:5.1f}m  [SKIP -- semua trial invalid, durasi_adaptif={dur_used:.2f}s]")
            rows.append(dict(D0=D0, n_valid=0, mean_est=np.nan, bias=np.nan,
                              std=np.nan, rmse=np.nan, rel_rmse_pct=np.nan,
                              duration_s=dur_used))
            continue
        bias = np.mean(estimates) - true_a
        std = np.std(estimates)
        rmse = np.sqrt(np.mean((estimates - true_a) ** 2))
        rel_rmse_pct = 100 * rmse / abs(true_a)
        rows.append(dict(D0=D0, n_valid=len(estimates), mean_est=np.mean(estimates),
                          bias=bias, std=std, rmse=rmse, rel_rmse_pct=rel_rmse_pct,
                          duration_s=dur_used))
        print(f"  D0={D0:5.1f}m  durasi={dur_used:.2f}s  mean_est={np.mean(estimates):+.3f}  "
              f"bias={bias:+.3f}  std={std:.3f}  rel_RMSE={rel_rmse_pct:5.1f}%")
    return pd.DataFrame(rows)


def recommend_effective_range(sweep_df: pd.DataFrame,
                               thresholds_pct=(25, 50, 100)) -> dict:
    """Cari D0 maksimum di mana rel_RMSE masih di bawah tiap ambang."""
    recs = {}
    sweep_sorted = sweep_df.sort_values("D0")
    for thr in thresholds_pct:
        ok = sweep_sorted[sweep_sorted["rel_rmse_pct"] <= thr]
        recs[thr] = float(ok["D0"].max()) if not ok.empty else None
    return recs


# =============================================================================
# Orkestrasi
# =============================================================================

if __name__ == "__main__":
    F_PX, W_REAL = 800.0, 1.8

    scenarios = [
        dict(sigma_f_px=2.0, sigma_W_m=0.05, sigma_w_px=2.5,
             label="controlled (W terukur langsung, sigma_W=0.05m)"),
        dict(sigma_f_px=2.0, sigma_W_m=0.15, sigma_w_px=2.5,
             label="naturalistic (W tak diketahui, sigma_W=0.15m)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    all_analytical = {}
    all_empirical = {}

    for sc in scenarios:
        D_grid, sigma_D, rel_pct, D_cross = run_analytical_analysis(
            f_px=F_PX, W_real_m=W_REAL, **sc)
        all_analytical[sc["label"]] = (D_grid, rel_pct, D_cross)

        axes[0].loglog(D_grid, sigma_D, label=sc["label"])
        axes[1].plot(D_grid, rel_pct, label=sc["label"])

    axes[0].set_xlabel("Jarak D (m, log)")
    axes[0].set_ylabel("sigma_D (m, log)")
    axes[0].set_title("Kurva error jarak (log-log)\n(perhatikan perubahan slope -- D^1 -> D^2)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].axhline(25, color="orange", linestyle="--", linewidth=0.8, label="25% (ketat)")
    axes[1].axhline(50, color="red", linestyle="--", linewidth=0.8, label="50% (longgar)")
    axes[1].set_xlabel("Jarak D (m)")
    axes[1].set_ylabel("sigma_D / D (%)")
    axes[1].set_title("Error jarak relatif vs jarak")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./effective_range_analytical.png", dpi=130)
    print("\nPlot analitik disimpan: effective_range_analytical.png")

    # ---- Empiris: hanya utk skenario "controlled" (lebih cepat dihitung,
    #      cukup utk validasi silang thd hasil analitik) ----
    D0_grid = [8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]
    sweep_df = run_empirical_sweep(
        D0_grid, true_a=-1.5, n_trials=12,
        f_px=F_PX, W_real_m=W_REAL, **scenarios[0])
    sweep_df.to_csv("./effective_range_empirical_sweep.csv", index=False)

    recs = recommend_effective_range(sweep_df)
    print("\n" + "=" * 70)
    print("REKOMENDASI EFFECTIVE RANGE (empiris, true_a=-1.5 m/s^2)")
    print("=" * 70)
    for thr, D_max in recs.items():
        print(f"  Toleransi rel-RMSE <= {thr:3d}% : jarak andal sampai "
              f"~{D_max if D_max else 'TIDAK ADA D0 memenuhi'} m")

    D_cross_controlled = all_analytical[scenarios[0]["label"]][2]
    print(f"\nBandingkan dgn D_cross analitik: {D_cross_controlled:.1f} m")
    print("(titik di mana noise piksel mulai mendominasi kalibrasi -- ini")
    print(" harus berada di kisaran yg sama dgn ambang 25-50% empiris di atas,")
    print(" krn di situlah error mulai naik tajam / non-linear)")

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sweep_df["D0"], sweep_df["mean_est"], yerr=sweep_df["std"],
                marker="o", capsize=4, label="Kalman -- mean +- std (12 trial)")
    ax.axhline(-1.5, color="green", linestyle="--", label="true_a = -1.5 m/s^2")
    ax.axvline(D_cross_controlled, color="red", linestyle=":",
               label=f"D_cross analitik = {D_cross_controlled:.1f} m")
    ax.set_xlabel("Jarak awal D0 (m)")
    ax.set_ylabel("Estimasi percepatan (m/s^2)")
    ax.set_title("Akurasi estimasi percepatan vs jarak (empiris, Kalman)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./effective_range_empirical.png", dpi=130)
    print("\nPlot empiris disimpan: effective_range_empirical.png")
