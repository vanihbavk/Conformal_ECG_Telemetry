"""Downstream clinical ECG metrics: Beat F1, ectopic recall, HRV RMSSD, and QRS PRD."""

from __future__ import annotations

import numpy as np

from .config import MetricsConfig

_DEFAULT_METRICS = MetricsConfig()


def match_beats_aami(
    ref_peaks_s: np.ndarray,
    est_peaks_s: np.ndarray,
    ect_flags: np.ndarray | None = None,
    tol_s: float | None = None,
    cfg: MetricsConfig | None = None,
) -> dict[str, float]:
    """One-to-one nearest-neighbor beat matching within +/-75 ms."""
    c = cfg or _DEFAULT_METRICS
    tol = c.match_tolerance_s if tol_s is None else tol_s
    n_ref = len(ref_peaks_s)
    n_est = len(est_peaks_s)
    if n_ref == 0 or n_est == 0:
        return {
            "tp": 0.0,
            "fp": float(n_est),
            "fn": float(n_ref),
            "beat_f1": 0.0,
            "sensitivity": 0.0,
            "ppv": 0.0,
            "ectopic_recall": np.nan,
        }

    dists = np.abs(ref_peaks_s[:, None] - est_peaks_s[None, :])
    rows, cols = np.where(dists <= tol)
    pairs = sorted((float(dists[r, col]), int(r), int(col)) for r, col in zip(rows, cols))

    matched_ref: set[int] = set()
    matched_est: set[int] = set()
    for _, r, col in pairs:
        if r not in matched_ref and col not in matched_est:
            matched_ref.add(r)
            matched_est.add(col)

    tp = len(matched_ref)
    fp = n_est - tp
    fn = n_ref - tp
    f1 = float((2.0 * tp) / (n_ref + n_est)) if (n_ref + n_est) > 0 else 0.0
    sens = float(tp / n_ref) if n_ref > 0 else 0.0
    ppv = float(tp / n_est) if n_est > 0 else 0.0

    ect_recall = np.nan
    if ect_flags is not None and np.any(ect_flags):
        ect_indices = np.where(ect_flags)[0]
        ect_hits = sum(1 for i in ect_indices if int(i) in matched_ref)
        ect_recall = float(ect_hits / len(ect_indices))

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "beat_f1": f1,
        "sensitivity": sens,
        "ppv": ppv,
        "ectopic_recall": ect_recall,
    }


def compute_rmssd_ms(peaks_s: np.ndarray) -> float:
    """RMSSD (ms) across physiological RR intervals."""
    if len(peaks_s) < 3:
        return np.nan
    rr = np.diff(np.sort(peaks_s))
    valid = rr[(rr >= 0.25) & (rr <= 2.0)]
    if len(valid) < 2:
        return np.nan
    diff_rr_ms = np.diff(valid) * 1000.0
    return float(np.sqrt(np.mean(diff_rr_ms * diff_rr_ms)))


def compute_qrs_morphology_prd(
    sig_hat_clin: np.ndarray,
    sig_clean_clin: np.ndarray,
    ref_peaks_s: np.ndarray,
    fs: float = 360.0,
) -> dict[str, float]:
    """Beat-aligned QRS PRD (%) and R-amplitude error (%) vs clean reference."""
    half_qrs = int(round(0.050 * fs))
    base_lo = int(round(0.110 * fs))
    base_hi = int(round(0.055 * fs))
    n = len(sig_clean_clin)
    sq_err = 0.0
    sq_ref = 0.0
    r_errs = []

    for r_s in ref_peaks_s:
        c = int(round(r_s * fs))
        if c - base_lo - 4 < 0 or c + half_qrs + 4 >= n:
            continue
        b_ref = float(np.median(sig_clean_clin[c - base_lo : c - base_hi + 1]))
        b_hat = float(np.median(sig_hat_clin[c - base_lo : c - base_hi + 1]))
        q_ref = sig_clean_clin[c - half_qrs : c + half_qrs + 1] - b_ref
        best_sse = np.inf
        best_q_hat = None
        for lag in (-2, -1, 0, 1, 2):
            q_hat = sig_hat_clin[c + lag - half_qrs : c + lag + half_qrs + 1] - b_hat
            sse = float(np.sum((q_hat - q_ref) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_q_hat = q_hat
        sq_err += best_sse
        sq_ref += float(np.sum(q_ref * q_ref))
        ref_pk = float(np.max(np.abs(q_ref)))
        if ref_pk > 0.05 and best_q_hat is not None:
            hat_pk = float(np.max(np.abs(best_q_hat)))
            r_errs.append(abs(hat_pk - ref_pk) / ref_pk * 100.0)

    if sq_ref <= 1e-9:
        return {"qrs_prd_pct": 100.0, "r_amp_err_pct": 100.0}
    prd = float(np.sqrt(sq_err / sq_ref) * 100.0)
    r_err = float(np.median(r_errs)) if r_errs else 100.0
    return {"qrs_prd_pct": prd, "r_amp_err_pct": r_err}


def evaluate_clinical_window(
    sig_hat_clin: np.ndarray,
    sig_clean_clin: np.ndarray,
    ref_peaks_s: np.ndarray,
    est_peaks_s: np.ndarray,
    ect_flags: np.ndarray | None = None,
    fs: float = 360.0,
    cfg: MetricsConfig | None = None,
) -> dict[str, float]:
    """Compute clinical endpoints and binary failure indicators."""
    c = cfg or _DEFAULT_METRICS
    beat_m = match_beats_aami(ref_peaks_s, est_peaks_s, ect_flags=ect_flags, tol_s=c.match_tolerance_s, cfg=c)
    rmssd_ref = compute_rmssd_ms(ref_peaks_s)
    rmssd_est = compute_rmssd_ms(est_peaks_s)
    if np.isfinite(rmssd_ref):
        if np.isfinite(rmssd_est):
            rmssd_err_ms = float(abs(rmssd_est - rmssd_ref))
            rmssd_rel = float(rmssd_err_ms / max(rmssd_ref, 20.0))
        else:
            rmssd_err_ms = 200.0
            rmssd_rel = 2.0
    else:
        rmssd_err_ms = 0.0
        rmssd_rel = 0.0

    morph = compute_qrs_morphology_prd(sig_hat_clin, sig_clean_clin, ref_peaks_s, fs=fs)

    fail_f1 = beat_m["beat_f1"] < c.min_beat_f1
    fail_hrv = (rmssd_err_ms > c.max_rmssd_ms) and (rmssd_rel > c.max_rmssd_rel)
    fail_prd = morph["qrs_prd_pct"] > c.max_qrs_prd
    clin_fail = float(fail_f1 or fail_hrv or fail_prd)
    clin_fail_det_hrv = float(fail_f1 or fail_hrv)

    fail_strict = float(
        (beat_m["beat_f1"] < 0.95) or ((rmssd_err_ms > 15.0) and (rmssd_rel > 0.15)) or (morph["qrs_prd_pct"] > 15.0)
    )
    fail_lenient = float(
        (beat_m["beat_f1"] < 0.85) or ((rmssd_err_ms > 30.0) and (rmssd_rel > 0.35)) or (morph["qrs_prd_pct"] > 35.0)
    )

    return {
        "beat_f1": beat_m["beat_f1"],
        "sensitivity": beat_m["sensitivity"],
        "ppv": beat_m["ppv"],
        "ectopic_recall": beat_m["ectopic_recall"],
        "rmssd_err_ms": rmssd_err_ms,
        "rmssd_rel_err": rmssd_rel,
        "qrs_prd_pct": morph["qrs_prd_pct"],
        "r_amp_err_pct": morph["r_amp_err_pct"],
        "clin_fail": clin_fail,
        "clin_fail_det_hrv": clin_fail_det_hrv,
        "clin_fail_strict": fail_strict,
        "clin_fail_lenient": fail_lenient,
    }
