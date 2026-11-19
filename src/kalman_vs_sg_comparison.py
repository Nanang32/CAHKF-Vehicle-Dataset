"""
kalman_vs_sg_comparison.py
============================
Uji Kalman filter (model constant-acceleration, "discrete white noise
acceleration" / DWNA -- Bar-Shalom et al.) sebagai pembanding Savitzky-Golay
untuk estimasi D(t)/v(t)/a(t) dari lebar bbox.

Kenapa Kalman relevan di sini:
  1. Menangani sampling TIDAK SERAGAM secara native lewat F(dt), Q(dt) --
     tidak perlu generalisasi ad-hoc seperti local-polynomial SG yang
     dipakai di test_sg_real_data.py.
  2. Tuning process-noise (q) bisa lewat minimisasi INNOVATION RMSE
     (galat prediksi satu-langkah-ke-depan, x_pred sebelum update memakai
     observasi saat ini) -- ini setara filosofis dengan LOO-RMSE yang
     dipakai memilih window SG, tapi native ke algoritma filter, bukan
     cross-validation terpisah.
  3. RTS smoother (backward pass) memberi estimasi post-hoc yang memakai
     SELURUH data sesi (bukan cuma window lokal k-tetangga), cocok untuk
     analisis pasca-sesi (bukan real-time).

Model state: x = [D, v, a]^T
  F(dt) = [[1, dt, dt^2/2],
           [0, 1,  dt],
           [0, 0,  1]]
  H = [1, 0, 0]      (kita hanya mengamati D, dari lebar bbox)
  Q(dt) = q * [[dt^4/4, dt^3/2, dt^2/2],
               [dt^3/2, dt^2,   dt],
               [dt^2/2, dt,     1]]
  R = varians noise pengukuran D (dari propagasi error kalibrasi f, W_real,
      dan noise deteksi bbox -- lihat sg_acceleration_estimation.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Kalman filter (constant-acceleration, DWNA) + RTS smoother
# ---------------------------------------------------------------------------

def _F(dt: float) -> np.ndarray:
    return np.array([[1, dt, 0.5 * dt ** 2],
                      [0, 1, dt],
                      [0, 0, 1]])


def _Q(dt: float, q: float) -> np.ndarray:
    return q * np.array([
        [dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
        [dt ** 3 / 2, dt ** 2,     dt],
        [dt ** 2 / 2, dt,          1.0],
    ])


H = np.array([[1.0, 0.0, 0.0]])


def kf_filter(t: np.ndarray, z: np.ndarray, q: float, r: float,
              x0: np.ndarray | None = None, P0: np.ndarray | None = None):
    """
    Forward Kalman filter pass. Mengembalikan state & kovarian ter-filter
    (x_filt, P_filt) DAN prior/predicted state sebelum update di tiap
    langkah (x_pred, P_pred) -- dipakai untuk innovation-based validation.
    """
    n = len(t)
    if x0 is None:
        x0 = np.array([z[0], 0.0, 0.0])
    if P0 is None:
        P0 = np.diag([r * 10, 100.0, 100.0])  # prior lebar krn belum ada info

    x = x0.copy()
    P = P0.copy()

    x_filt = np.zeros((n, 3))
    P_filt = np.zeros((n, 3, 3))
    x_pred_arr = np.zeros((n, 3))
    P_pred_arr = np.zeros((n, 3, 3))
    innovations = np.full(n, np.nan)

    x_filt[0], P_filt[0] = x0, P0
    x_pred_arr[0], P_pred_arr[0] = x0, P0

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        F = _F(dt)
        Q = _Q(dt, q)

        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        x_pred_arr[k], P_pred_arr[k] = x_pred, P_pred

        if np.isnan(z[k]):
            # tidak ada observasi frame ini (mis. objek sedang off-frame
            # dlm grace period) -- lewati update, cuma propagate
            x, P = x_pred, P_pred
        else:
            # update
            S = H @ P_pred @ H.T + r          # innovation covariance (scalar)
            innovation = z[k] - (H @ x_pred)[0]
            innovations[k] = innovation
            K = (P_pred @ H.T) / S            # Kalman gain (3x1)
            x = x_pred + (K.flatten() * innovation)
            P = P_pred - K @ H @ P_pred

        x_filt[k], P_filt[k] = x, P

    return dict(x_filt=x_filt, P_filt=P_filt, x_pred=x_pred_arr,
                P_pred=P_pred_arr, innovations=innovations)


def rts_smoother(t: np.ndarray, filt_result: dict, q: float) -> np.ndarray:
    """Rauch-Tung-Striebel backward pass -- estimasi memakai SELURUH data
    sesi, dipakai untuk analisis pasca-sesi (bukan real-time)."""
    n = len(t)
    x_filt, P_filt = filt_result["x_filt"], filt_result["P_filt"]

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F = _F(dt)
        Q = _Q(dt, q)
        P_pred_next = F @ P_filt[k] @ F.T + Q

        # regularisasi kecil agar invers stabil jika P_pred_next singular
        C = P_filt[k] @ F.T @ np.linalg.pinv(P_pred_next)

        x_pred_next = F @ x_filt[k]
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred_next)
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred_next) @ C.T

    return x_smooth


def innovation_rmse(filt_result: dict) -> float:
    """Analog Kalman utk LOO-RMSE di SG: galat prediksi satu-langkah-ke-depan
    (x_pred SEBELUM melihat observasi saat ini) -- murni out-of-sample."""
    innov = filt_result["innovations"]
    return float(np.sqrt(np.nanmean(innov ** 2)))


def tune_process_noise(t: np.ndarray, z: np.ndarray, r: float,
                        q_grid=None) -> tuple[float, dict]:
    """Grid search q yang meminimalkan innovation RMSE -- setara filosofis
    dgn pemilihan window k di SG via LOO-RMSE, tapi native ke filter."""
    if q_grid is None:
        q_grid = np.logspace(-3, 3, 25)
    best_q, best_rmse, best_result = None, np.inf, None
    for q in q_grid:
        result = kf_filter(t, z, q=q, r=r)
        rmse = innovation_rmse(result)
        if rmse < best_rmse:
            best_q, best_rmse, best_result = q, rmse, result
    return best_q, best_result


# ---------------------------------------------------------------------------
# 2. SG (local-polynomial, non-uniform) -- direplikasi ringkas dari file
#    sebelumnya agar skrip ini berdiri sendiri untuk perbandingan langsung.
# ---------------------------------------------------------------------------

def local_poly_fit(t, y, t0, k, poly_order=2, exclude_center=False):
    idx_sorted = np.argsort(np.abs(t - t0))
    if exclude_center:
        idx_sorted = idx_sorted[t[idx_sorted] != t0]
    idx = idx_sorted[:k]
    if len(idx) < poly_order + 1:
        return np.nan, np.nan, np.nan
    dt = t[idx] - t0
    A = np.vander(dt, poly_order + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
    return coef[0], coef[1], 2 * coef[2]


def sg_loo_rmse(t, y, k, poly_order=2):
    n = len(t)
    pred = np.full(n, np.nan)
    for i in range(n):
        v, _, _ = local_poly_fit(t, y, t[i], k, poly_order, exclude_center=True)
        pred[i] = v
    return float(np.sqrt(np.nanmean((pred - y) ** 2)))


def sg_best_window(t, y, k_candidates=(5, 7, 9, 11), poly_order=2):
    scores = {k: sg_loo_rmse(t, y, k, poly_order) for k in k_candidates
              if len(t) >= k + poly_order}
    best_k = min(scores, key=scores.get)
    return best_k, scores


def sg_apply(t, y, k, poly_order=2):
    n = len(t)
    val = np.full(n, np.nan)
    vel = np.full(n, np.nan)
    acc = np.full(n, np.nan)
    for i in range(n):
        val[i], vel[i], acc[i] = local_poly_fit(t, y, t[i], k, poly_order)
    return val, vel, acc


# ---------------------------------------------------------------------------
# 3. Uji A: data sintetis maneuver terkontrol (sama seperti postsession_analysis.py)
# ---------------------------------------------------------------------------

def _synthetic_trial(true_a, D0=30.0, v0=-1.5, fps=15, duration_s=4.0,
                      f_px=800.0, W_real=1.8, noise_px=2.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    D_true = D0 + v0 * t + 0.5 * true_a * t ** 2
    w_true = (f_px * W_real) / D_true
    w_noisy = w_true + rng.normal(0, noise_px, n)
    D_meas = (f_px * W_real) / w_noisy
    return t, D_meas, D_true


def test_synthetic_maneuvers():
    print("=" * 78)
    print("UJI A: Kalman vs SG pada data sintetis maneuver terkontrol")
    print("(noise_px=2.5 -- level noise 'realistis' yang bikin SG kewalahan")
    print(" di demo postsession_analysis.py sebelumnya)")
    print("=" * 78)

    maneuvers = {"gas": 2.0, "rem": -2.0, "konstan": 0.0}
    n_trials = 8
    r_meas = 0.5 ** 2  # asumsi varians noise D awal (m^2), dari propagasi kalibrasi

    rows = []
    for maneuver, true_a in maneuvers.items():
        for trial in range(n_trials):
            t, D_meas, D_true = _synthetic_trial(true_a, seed=trial * 17 + {'gas': 101, 'rem': 202, 'konstan': 303}[maneuver])
            marker_mid = 1.1  # detik, sama seperti demo sebelumnya

            # --- SG ---
            best_k, _ = sg_best_window(t, D_meas)
            _, _, acc_sg_arr = sg_apply(t, D_meas, best_k)
            sg_marker_accel, _, _ = local_poly_fit(t, D_meas, marker_mid, best_k)
            _, _, sg_marker_accel = local_poly_fit(t, D_meas, marker_mid, best_k)

            # --- Kalman ---
            best_q, filt = tune_process_noise(t, D_meas, r=r_meas)
            x_smooth = rts_smoother(t, filt, q=best_q)
            idx_marker = np.argmin(np.abs(t - marker_mid))
            kf_marker_accel = x_smooth[idx_marker, 2]
            kf_innov_rmse = innovation_rmse(filt)

            rows.append(dict(
                maneuver=maneuver, trial=trial, true_a=true_a,
                sg_best_k=best_k, sg_loo_rmse=sg_best_window(t, D_meas)[1][best_k],
                sg_marker_accel=sg_marker_accel,
                kf_best_q=best_q, kf_innov_rmse=kf_innov_rmse,
                kf_marker_accel=kf_marker_accel,
            ))

    df = pd.DataFrame(rows)
    df.to_csv("./kalman_vs_sg_synthetic_results.csv", index=False)

    print("\nPer-trial (10 baris pertama):")
    print(df.head(10).to_string(index=False))

    summary = df.groupby("maneuver").agg(
        true_a=("true_a", "first"),
        sg_mean=("sg_marker_accel", "mean"), sg_std=("sg_marker_accel", "std"),
        kf_mean=("kf_marker_accel", "mean"), kf_std=("kf_marker_accel", "std"),
    ).reset_index()
    summary["sg_bias"] = summary["sg_mean"] - summary["true_a"]
    summary["kf_bias"] = summary["kf_mean"] - summary["true_a"]

    print("\n" + "=" * 78)
    print("RINGKASAN (n=8 trial/maneuver) -- bandingkan bias & std terhadap true_a")
    print("=" * 78)
    print(summary.to_string(index=False))
    summary.to_csv("./kalman_vs_sg_synthetic_summary.csv", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# 4. Uji B: data RIIL event-based (internal_id=4, sampling sangat tidak
#    seragam -- kasus di mana keunggulan Kalman paling relevan)
# ---------------------------------------------------------------------------

def test_real_irregular_track():
    print("\n" + "=" * 78)
    print("UJI B: Kalman vs SG pada track RIIL (internal_id=4, event-based,")
    print("sampling gap median 0.34s, maksimum 11.6s)")
    print("=" * 78)

    import openpyxl
    wb = openpyxl.load_workbook(
        "/mnt/user-data/uploads/UJI_HASIL_statistik_tracking_log.xlsx", data_only=True)
    ws = wb["tracking_log_20260428_114926"]
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["lebar_mm"] = pd.to_numeric(df["lebar_mm"], errors="coerce")
    df["waktu_unix"] = pd.to_numeric(df["waktu_unix"], errors="coerce")

    sub = df[(df["internal_id"] == 4) & df["lebar_mm"].notna()].copy()
    sub = sub.sort_values("waktu_unix").drop_duplicates(subset="waktu_unix")
    t = (sub["waktu_unix"].values - sub["waktu_unix"].values[0]) / 1000.0
    D_meas = sub["lebar_mm"].values.astype(float)  # proxy (bukan mm jarak sebenarnya)

    # SG terbaik (dari analisis sebelumnya, k=9 adalah titik keseimbangan)
    best_k, sg_scores = sg_best_window(t, D_meas)
    print(f"SG: window terbaik k={best_k}, LOO-RMSE per window: "
          f"{ {k: round(v,2) for k,v in sg_scores.items()} }")

    # Kalman: tuning q via innovation RMSE, R diestimasi dari noise lokal
    r_est = float(np.nanvar(np.diff(D_meas))) / 2  # estimasi kasar var noise
    best_q, filt = tune_process_noise(t, D_meas, r=r_est)
    kf_rmse = innovation_rmse(filt)
    print(f"Kalman: q terbaik={best_q:.4g}, innovation RMSE={kf_rmse:.2f} "
          f"(vs SG LOO-RMSE k={best_k}: {sg_scores[best_k]:.2f})")

    print("\nCatatan: pada data event-based dgn gap ekstrem (11.6s), Kalman")
    print("secara native menangani lewat F(dt)/Q(dt) yang scale dgn dt -- SG")
    print("local-polynomial jg bisa, tapi Kalman TIDAK PERLU pemilihan k")
    print("terpisah dari 'jumlah tetangga' yg jadi kabur artinya saat gap")
    print("antar tetangga sangat bervariasi (5 tetangga bisa berarti 0.5s")
    print("atau 50s tergantung lokasi gap).")


if __name__ == "__main__":
    test_synthetic_maneuvers()
    test_real_irregular_track()
