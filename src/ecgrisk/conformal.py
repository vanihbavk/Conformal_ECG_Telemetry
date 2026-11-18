"""Patient-cluster Conformal Risk Control (CRC), Mondrian calibration, and cluster bootstrap CIs."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .config import ConformalConfig

_DEFAULT_CONFORMAL = ConformalConfig()


def fit_robust_score_params(
    features: np.ndarray,
    losses_target: np.ndarray | None = None,
    cfg: ConformalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median, IQR (q75 - q25), and L1-normalized point-biserial weights."""
    c = cfg or _DEFAULT_CONFORMAL
    med = np.median(features, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    z = np.clip((features - med[None, :]) / scale[None, :], 0.0, c.z_clip_max)
    if losses_target is None or np.std(losses_target) < 1e-8:
        w = np.ones(features.shape[1], dtype=float) / float(features.shape[1])
        return med, scale, w

    y = (losses_target - np.mean(losses_target)) / max(float(np.std(losses_target)), 1e-6)
    corrs = np.zeros(features.shape[1], dtype=float)
    for j in range(features.shape[1]):
        sd_j = float(np.std(z[:, j]))
        if sd_j > 1e-8:
            corrs[j] = float(np.mean(((z[:, j] - np.mean(z[:, j])) / sd_j) * y))
    w = np.maximum(corrs, c.min_weight)
    w = w / float(np.sum(w))
    return med, scale, w


def score_uncertainty(
    features: np.ndarray,
    med: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    cfg: ConformalConfig | None = None,
) -> np.ndarray:
    """Compute monotone risk score S(x) = sum_j w_j * clip((f_j - med_j) / scale_j, 0, z_clip_max)."""
    c = cfg or _DEFAULT_CONFORMAL
    z = np.clip((features - med[None, :]) / scale[None, :], 0.0, c.z_clip_max)
    return z @ weights


def score_unsupervised(
    features: np.ndarray,
    cfg: ConformalConfig | None = None,
) -> np.ndarray:
    """Compute zero-training fixed-scale equal-weight risk score S_0(x)."""
    c = cfg or _DEFAULT_CONFORMAL
    med = np.zeros(features.shape[1], dtype=float)
    scales = np.asarray(c.fixed_unsup_scales[: features.shape[1]], dtype=float)
    weights = np.asarray(c.fixed_unsup_weights[: features.shape[1]], dtype=float)
    weights = weights / float(np.sum(weights))
    return score_uncertainty(features, med, scales, weights, cfg=c)


def calibrate_iid_threshold(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    alpha: float,
) -> float:
    """Window-level i.i.d. conformal threshold."""
    n = len(cal_scores)
    if n == 0 or (1.0 / (n + 1.0)) > alpha:
        return -np.inf
    low_sentinel = float(np.min(cal_scores)) - 1e-9
    cand = np.unique(np.concatenate([[low_sentinel], cal_scores]))

    lo, hi, best_idx = 0, len(cand) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        tau = cand[mid]
        win_loss = np.where(cal_scores <= tau, cal_losses, 0.0)
        if (float(np.sum(win_loss)) + 1.0) / float(n + 1) <= alpha + 1e-12:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return float(cand[best_idx]) if best_idx >= 0 else -np.inf


def calibrate_crc_threshold(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    cal_patient_ids: np.ndarray,
    alpha: float,
) -> float:
    """Patient-cluster CRC threshold satisfying (sum_p L_p(tau) + 1) / (M + 1) <= alpha."""
    unique_patients = np.unique(cal_patient_ids)
    m_patients = len(unique_patients)
    if m_patients == 0 or (1.0 / (m_patients + 1.0)) > alpha:
        return -np.inf

    low_sentinel = float(np.min(cal_scores)) - 1e-9
    cand = np.unique(np.concatenate([[low_sentinel], cal_scores]))
    patient_masks = [cal_patient_ids == pid for pid in unique_patients]

    lo, hi, best_idx = 0, len(cand) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        tau = cand[mid]
        win_loss = np.where(cal_scores <= tau, cal_losses, 0.0)
        sum_patient_risk = sum(float(np.mean(win_loss[pm])) for pm in patient_masks)
        if (sum_patient_risk + 1.0) / float(m_patients + 1) <= alpha + 1e-12:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return float(cand[best_idx]) if best_idx >= 0 else -np.inf


def calibrate_group_crc_thresholds(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    cal_patient_ids: np.ndarray,
    cal_groups: np.ndarray,
    alpha: float,
) -> dict[str, dict[str, float | bool | int]]:
    """Mondrian (subgroup-conditional) Patient-Cluster CRC thresholds per group."""
    out: dict[str, dict[str, float | bool | int]] = {}
    for g in sorted(np.unique(cal_groups)):
        g_mask = cal_groups == g
        m_g = int(len(np.unique(cal_patient_ids[g_mask])))
        min_alpha = 1.0 / float(m_g + 1)
        tau_g = calibrate_crc_threshold(
            cal_scores[g_mask],
            cal_losses[g_mask],
            cal_patient_ids[g_mask],
            alpha=alpha,
        )
        out[str(g)] = {
            "tau": float(tau_g),
            "m_patients": m_g,
            "min_feasible_alpha": float(min_alpha),
            "feasible": bool(min_alpha <= alpha + 1e-12),
        }
    return out


def evaluate_gating_decision(
    accept: np.ndarray,
    losses: np.ndarray,
    beat_f1: np.ndarray,
    rmssd_err_ms: np.ndarray,
    qrs_prd_pct: np.ndarray,
    ectopic_recall: np.ndarray,
) -> dict[str, float]:
    """Compute acceptance %, silent risk, conditional failure %, and accepted quality medians."""
    accept = np.asarray(accept, dtype=bool)
    silent_risk = float(np.mean(np.where(accept, losses, 0.0)))
    accept_pct = float(np.mean(accept)) * 100.0
    abstain_pct = 100.0 - accept_pct
    cond_fail = float(np.mean(losses[accept])) * 100.0 if np.any(accept) else 0.0

    acc_f1_med = float(np.median(beat_f1[accept])) if np.any(accept) else np.nan
    acc_rmssd_med = float(np.median(rmssd_err_ms[accept])) if np.any(accept) else np.nan
    acc_prd_med = float(np.median(qrs_prd_pct[accept])) if np.any(accept) else np.nan

    ect_valid = accept & np.isfinite(ectopic_recall)
    acc_ect_rec = float(np.mean(ectopic_recall[ect_valid])) * 100.0 if np.any(ect_valid) else np.nan
    all_ect_valid = np.isfinite(ectopic_recall)
    ect_win_accept = float(np.mean(accept[all_ect_valid])) * 100.0 if np.any(all_ect_valid) else np.nan

    return {
        "accept_pct": accept_pct,
        "abstain_pct": abstain_pct,
        "silent_risk": silent_risk,
        "cond_fail_pct": cond_fail,
        "acc_beat_f1_med": acc_f1_med,
        "acc_rmssd_err_med": acc_rmssd_med,
        "acc_qrs_prd_med": acc_prd_med,
        "acc_ect_recall_pct": acc_ect_rec,
        "ect_win_accept_pct": ect_win_accept,
    }


def patient_cluster_bootstrap_ci(
    values_by_patient: list[float] | np.ndarray,
    n_boot: int = 1000,
    seed: int = 123,
) -> tuple[float, float, float]:
    """Mean and 95% cluster-bootstrap CI across patients."""
    arr = np.asarray(values_by_patient, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def wilson_binomial_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """95% Wilson score interval for k / n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / float(n))) / denom
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def patient_clustered_validity_test(
    patient_risks: np.ndarray | list[float],
    alpha: float,
) -> dict[str, float]:
    """One-sided patient-level t-test for H0: E[L_test] <= alpha vs H1: E[L_test] > alpha."""
    arr = np.asarray(patient_risks, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr)) if n == 1 else 0.0
        return {"mean": m, "se": 0.0, "ci_lo": m, "ci_hi": m, "t_stat": 0.0, "p_viol": 0.5}

    mean = float(np.mean(arr))
    se = float(np.sqrt(np.var(arr, ddof=1) / float(n)))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    t_stat = (mean - alpha) / max(se, 1e-12)
    p_viol = float(1.0 - stats.t.cdf(t_stat, df=n - 1))

    return {
        "mean": mean,
        "se": se,
        "ci_lo": max(0.0, mean - t_crit * se),
        "ci_hi": mean + t_crit * se,
        "t_stat": float(t_stat),
        "p_viol": p_viol,
    }


def simulate_clustered_correlation_sweep(
    persistence_grid: tuple[float, ...] = (0.0, 0.40, 0.75, 0.90, 0.95),
    m_cal: int = 39,
    m_test: int = 50,
    n_trials: int = 300,
    alpha: float = 0.08,
    seed: int = 2026,
) -> list[dict[str, float | str]]:
    """Monte Carlo sweep of intra-patient correlation vs calibration validity."""
    rng = np.random.default_rng(seed)
    regimes: list[tuple[str, bool, float, bool]] = [("iid_homog", False, 0.0, False)]
    for rho in persistence_grid:
        regimes.append((f"clustered_rho_{rho:.2f}", True, float(rho), True))

    rows: list[dict[str, float | str]] = []
    for name, hetero, rho, var_len in regimes:
        crc_risks: list[float] = []
        iid_risks: list[float] = []
        crc_accs: list[float] = []
        iid_accs: list[float] = []

        for _ in range(n_trials):
            m_tot = m_cal + m_test
            duties = rng.beta(1.2, 2.5, size=m_tot) if hetero else np.full(m_tot, 0.28)
            all_s: list[np.ndarray] = []
            all_l: list[np.ndarray] = []

            for p in range(m_tot):
                d_p = float(duties[p])
                nw = int(rng.choice([30, 60, 120])) if var_len else 60
                p_11 = float(rho) if rho > 0.0 else d_p
                p_01 = min(0.95, d_p * (1.0 - p_11) / max(1.0 - d_p, 1e-3)) if rho > 0.0 else d_p
                state = bool(rng.random() < d_p)
                ps = np.empty(nw, dtype=float)
                pl = np.empty(nw, dtype=float)
                for w in range(nw):
                    if state:
                        fail = float(rng.random() < 0.75)
                        sc = float(rng.normal(1.25 if fail > 0.5 else 0.70, 0.45))
                    else:
                        fail = float(rng.random() < 0.04)
                        sc = float(rng.normal(1.10 if fail > 0.5 else 0.30, 0.25))
                    ps[w] = max(0.0, sc)
                    pl[w] = fail
                    state = bool(rng.random() < (p_11 if state else p_01))
                all_s.append(ps)
                all_l.append(pl)

            cal_s = np.concatenate(all_s[:m_cal])
            cal_l = np.concatenate(all_l[:m_cal])
            cal_p = np.concatenate([np.full(len(all_s[p]), p) for p in range(m_cal)])

            tau_crc = calibrate_crc_threshold(cal_s, cal_l, cal_p, alpha=alpha)
            tau_iid = calibrate_iid_threshold(cal_s, cal_l, alpha=alpha)

            pat_r_c = [float(np.mean(np.where(all_s[p] <= tau_crc, all_l[p], 0.0))) for p in range(m_cal, m_tot)]
            pat_r_i = [float(np.mean(np.where(all_s[p] <= tau_iid, all_l[p], 0.0))) for p in range(m_cal, m_tot)]
            pat_a_c = [float(np.mean(all_s[p] <= tau_crc)) * 100.0 for p in range(m_cal, m_tot)]
            pat_a_i = [float(np.mean(all_s[p] <= tau_iid)) * 100.0 for p in range(m_cal, m_tot)]

            crc_risks.append(float(np.mean(pat_r_c)))
            iid_risks.append(float(np.mean(pat_r_i)))
            crc_accs.append(float(np.mean(pat_a_c)))
            iid_accs.append(float(np.mean(pat_a_i)))

        crc_arr = np.asarray(crc_risks)
        iid_arr = np.asarray(iid_risks)
        rows.append(
            {
                "regime": name,
                "hetero": bool(hetero),
                "rho": float(rho),
                "crc_risk_mean_pct": float(np.mean(crc_arr)) * 100.0,
                "crc_risk_p95_pct": float(np.percentile(crc_arr, 95)) * 100.0,
                "crc_viol_pct": float(np.mean(crc_arr > alpha)) * 100.0,
                "crc_accept_pct": float(np.mean(crc_accs)),
                "iid_risk_mean_pct": float(np.mean(iid_arr)) * 100.0,
                "iid_risk_p95_pct": float(np.percentile(iid_arr, 95)) * 100.0,
                "iid_viol_pct": float(np.mean(iid_arr > alpha)) * 100.0,
                "iid_accept_pct": float(np.mean(iid_accs)),
            }
        )
    return rows


def evaluate_endpoint_distributions(
    accepted_values: np.ndarray,
) -> dict[str, float]:
    """Quantiles (p10, p25, median, p75, p90) and min/max for accepted windows."""
    arr = np.asarray(accepted_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"p10": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p90": np.nan, "min": np.nan, "max": np.nan}
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
