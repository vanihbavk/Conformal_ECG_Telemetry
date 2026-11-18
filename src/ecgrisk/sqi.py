"""Zero-oracle QRS detection, two-template morphological dispersion, and SQI extractors."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt, welch

from .config import SqiConfig

_DEFAULT_SQI = SqiConfig()


def _bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, fs: float, order: int = 3) -> np.ndarray:
    sos = butter(order, [lo_hz, min(hi_hz, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, x.astype(float))


def _refine_r_peaks(sig_ref: np.ndarray, cand_idx: np.ndarray, fs: float, half_ms: float = 40.0, min_dist_s: float = 0.20) -> np.ndarray:
    """Snap candidate peak indices to local dominant absolute deflection."""
    if len(cand_idx) == 0:
        return cand_idx
    half_s = max(1, int(round(half_ms * 1e-3 * fs)))
    n = len(sig_ref)
    refined = []
    min_dist = max(1, int(round(min_dist_s * fs)))
    for c in cand_idx:
        lo = max(0, int(c) - half_s)
        hi = min(n, int(c) + half_s + 1)
        seg = sig_ref[lo:hi]
        loc_base = float(np.median(seg))
        idx = lo + int(np.argmax(np.abs(seg - loc_base)))
        if not refined or (idx - refined[-1]) >= min_dist:
            refined.append(idx)
    return np.asarray(refined, dtype=int)


def _robust_chunk_median_max(sig_env: np.ndarray, fs: float, chunk_s: float = 1.5) -> float:
    """Median of contiguous block maxima."""
    c_len = max(16, int(round(chunk_s * fs)))
    n_blocks = len(sig_env) // c_len
    if n_blocks < 2:
        return float(np.percentile(sig_env, 95))
    blocks = sig_env[: n_blocks * c_len].reshape(n_blocks, c_len)
    return float(np.median(np.max(blocks, axis=1)))


def detect_qrs_pantompkins(sig_clin: np.ndarray, fs: float = 360.0, cfg: SqiConfig | None = None) -> np.ndarray:
    """Causal Pan-Tompkins QRS detector with RR-gap searchback."""
    c = cfg or _DEFAULT_SQI
    bp = _bandpass(sig_clin, c.pt_bp_low, c.pt_bp_high, fs=fs, order=2)
    d = np.zeros_like(bp)
    if len(bp) > 4:
        d[4:] = (2.0 * bp[4:] + bp[3:-1] - bp[1:-3] - 2.0 * bp[:-4]) / 8.0
    sq = d * d
    win = max(3, int(round(c.pt_mwi_s * fs)))
    mwi = np.convolve(sq, np.ones(win, dtype=float) / float(win), mode="same")

    skip = min(len(mwi) // 5, int(c.pt_refractory_s * fs))
    ref = mwi[skip:] if len(mwi) > skip + 20 else mwi
    noise_floor = float(np.percentile(ref, c.pt_thresh_pct))
    sig_scale = _robust_chunk_median_max(ref, fs=fs, chunk_s=1.5)
    thr_hi = max(noise_floor + c.pt_thresh_min * max(sig_scale - noise_floor, 1e-12), 1e-9)

    min_dist = max(1, int(round(c.pt_min_dist_s * fs)))
    peaks, _ = find_peaks(mwi, height=thr_hi, distance=min_dist)

    if len(peaks) >= 3:
        med_rr = float(np.median(np.diff(peaks)))
        thr_lo = max(noise_floor + 0.06 * max(sig_scale - noise_floor, 1e-12), 1e-9)
        extra = []
        for k in range(len(peaks) - 1):
            gap = peaks[k + 1] - peaks[k]
            if gap > 1.60 * med_rr:
                lo_i = int(peaks[k] + min_dist)
                hi_i = int(peaks[k + 1] - min_dist)
                if hi_i > lo_i + 2:
                    sub = mwi[lo_i:hi_i]
                    rel_pk = int(np.argmax(sub))
                    if sub[rel_pk] >= thr_lo:
                        extra.append(lo_i + rel_pk)
        if extra:
            peaks = np.sort(np.concatenate([peaks, np.asarray(extra, dtype=int)]))

    shifted = np.clip(peaks - win // 3, 0, len(sig_clin) - 1)
    refined = _refine_r_peaks(sig_clin, shifted, fs=fs, half_ms=40.0, min_dist_s=c.pt_refractory_s)
    return refined / float(fs)


def detect_qrs_steepness(sig_clin: np.ndarray, fs: float = 360.0, cfg: SqiConfig | None = None) -> np.ndarray:
    """Secondary steepness-envelope QRS detector."""
    c = cfg or _DEFAULT_SQI
    bp = _bandpass(sig_clin, c.steep_bp_low, c.steep_bp_high, fs=fs, order=3)
    slope = np.abs(np.diff(bp, prepend=bp[0]))
    win = max(3, int(round(c.steep_env_s * fs)))
    env = np.convolve(slope, np.ones(win, dtype=float) / float(win), mode="same")

    skip = min(len(env) // 5, int(c.pt_refractory_s * fs))
    ref = env[skip:] if len(env) > skip + 20 else env
    p25 = float(np.percentile(ref, c.steep_thresh_pct))
    sig_scale = _robust_chunk_median_max(ref, fs=fs, chunk_s=1.5)
    thr = max(p25 + c.steep_thresh_min * max(sig_scale - p25, 1e-12), 1e-6)

    min_dist = max(1, int(round(c.steep_min_dist_s * fs)))
    peaks, _ = find_peaks(env, height=thr, distance=min_dist)
    shifted = np.clip(peaks - win // 4, 0, len(sig_clin) - 1)
    refined = _refine_r_peaks(sig_clin, shifted, fs=fs, half_ms=40.0, min_dist_s=c.pt_refractory_s)
    return refined / float(fs)


def compute_bsqi(
    peaks_a_s: np.ndarray,
    peaks_b_s: np.ndarray,
    tol_s: float | None = None,
    cfg: SqiConfig | None = None,
) -> float:
    """Beat agreement F1 between two detectors within tol_s."""
    c = cfg or _DEFAULT_SQI
    tol = c.bsqi_tol_s if tol_s is None else tol_s
    if len(peaks_a_s) == 0 or len(peaks_b_s) == 0:
        return 0.0
    used_b: set[int] = set()
    matches = 0
    for ta in peaks_a_s:
        diffs = np.abs(peaks_b_s - ta)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol and j not in used_b:
            matches += 1
            used_b.add(j)
    return float((2.0 * matches) / (len(peaks_a_s) + len(peaks_b_s)))


def _extract_beat_snippets(
    sig_clin: np.ndarray,
    peaks_s: np.ndarray,
    fs: float = 360.0,
    half_qrs_ms: float = 90.0,
) -> np.ndarray:
    half_s = int(round(half_qrs_ms * 1e-3 * fs))
    pr_lo = int(round(0.120 * fs))
    pr_hi = int(round(0.060 * fs))
    n = len(sig_clin)
    snips = []
    for t_s in peaks_s:
        c = int(round(t_s * fs))
        if c - max(half_s, pr_lo) < 0 or c + half_s + 1 > n:
            continue
        base = float(np.median(sig_clin[c - pr_lo : c - pr_hi + 1]))
        seg = sig_clin[c - half_s : c + half_s + 1] - base
        norm = float(np.linalg.norm(seg))
        if norm > 1e-6:
            snips.append(seg / norm)
    if not snips:
        return np.zeros((0, 2 * half_s + 1), dtype=float)
    return np.asarray(snips, dtype=float)


def _huber_template(X: np.ndarray, n_iters: int = 5, delta: float = 0.25) -> np.ndarray:
    """Unit-norm Huber-IRLS template from L2-normalized beat snippets."""
    t = np.median(X, axis=0)
    tn = float(np.linalg.norm(t))
    if tn <= 1e-9:
        return t
    t = t / tn
    for _ in range(n_iters):
        res = np.linalg.norm(X - t[None, :], axis=1)
        w = np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-6))
        t_new = np.sum(w[:, None] * X, axis=0)
        tn = float(np.linalg.norm(t_new))
        if tn <= 1e-9:
            break
        t = t_new / tn
    return t


def compute_two_template_morph_disp(
    sig_clin: np.ndarray,
    peaks_s: np.ndarray,
    fs: float = 360.0,
    cfg: SqiConfig | None = None,
) -> tuple[float, float]:
    """Return (two_template_morph_disp, single_template_mean_corr)."""
    c = cfg or _DEFAULT_SQI
    X = _extract_beat_snippets(sig_clin, peaks_s, fs=fs, half_qrs_ms=c.morph_half_window_s * 1000.0)
    if len(X) < 3:
        return 1.0, 0.0

    t1 = _huber_template(X)
    sim1 = np.clip(X @ t1, -1.0, 1.0)
    single_mean_corr = float(np.mean(sim1))

    out_mask = sim1 < c.morph_cluster_corr_thresh
    if int(np.sum(out_mask)) >= c.morph_min_cluster_beats:
        t2 = _huber_template(X[out_mask])
        sim2 = np.clip(X @ t2, -1.0, 1.0)
        if float(np.mean(sim2[out_mask])) >= c.morph_secondary_coherence_min:
            best_sim = np.maximum(sim1, sim2)
        else:
            best_sim = sim1
    else:
        best_sim = sim1

    if len(peaks_s) < c.morph_min_beats:
        return float(c.morph_outlier_penalty), single_mean_corr
    morph_disp = float(np.clip(1.0 - float(np.mean(best_sim)), 0.0, 1.0))
    return morph_disp, single_mean_corr


def extract_sqi_features(
    sig_afe_150: np.ndarray,
    sig_clin_40: np.ndarray,
    fs: float = 360.0,
    cfg: SqiConfig | None = None,
) -> dict[str, object]:
    """Extract 5 zero-oracle SQI features and literature heuristic decisions."""
    c = cfg or _DEFAULT_SQI
    peaks_a = detect_qrs_pantompkins(sig_clin_40, fs=fs, cfg=c)
    peaks_b = detect_qrs_steepness(sig_clin_40, fs=fs, cfg=c)
    bsqi = compute_bsqi(peaks_a, peaks_b, tol_s=c.bsqi_tol_s, cfg=c)
    miss_bsqi = float(1.0 - bsqi)

    morph_disp, single_corr = compute_two_template_morph_disp(sig_clin_40, peaks_a, fs=fs, cfg=c)

    nperseg = min(len(sig_afe_150), int(2.0 * fs))
    freqs, psd_afe = welch(sig_afe_150, fs=fs, nperseg=nperseg)
    _, psd_clin = welch(sig_clin_40, fs=fs, nperseg=nperseg)

    tot_clin = float(np.sum(psd_clin[(freqs >= 0.5) & (freqs <= 40.0)])) + 1e-12
    p_low = float(np.sum(psd_clin[(freqs >= 0.0) & (freqs < c.bas_cutoff_hz)]))
    bas_sqi = float(np.clip(1.0 - p_low / (tot_clin + p_low + 1e-12), 0.0, 1.0))
    bas_noise = float(1.0 - bas_sqi)

    p_hf = float(np.sum(psd_afe[(freqs > c.hf_noise_low) & (freqs <= c.hf_noise_high)]))
    p_qrs = float(np.sum(psd_afe[(freqs >= c.hf_ref_low) & (freqs <= c.hf_ref_high)])) + 1e-9
    hf_noise = float(np.log1p(p_hf / p_qrs))

    xc = sig_clin_40 - np.median(sig_clin_40)
    sd = float(np.std(xc))
    kurt = 0.0 if sd < 1e-5 else float(np.mean((xc / sd) ** 4) - 3.0)
    kurt_deficit = float(np.clip((c.kurt_target - kurt) / c.kurt_scale, 0.0, c.kurt_clip_max))

    orph_ok = False
    if len(peaks_a) >= c.morph_min_beats:
        rr = np.diff(peaks_a)
        hr_bpm = 60.0 / max(float(np.median(rr)), 1e-3)
        rr_max = float(np.max(rr))
        rr_min = max(float(np.min(rr)), 1e-3)
        if (40.0 <= hr_bpm <= 180.0) and (rr_max <= 3.0) and ((rr_max / rr_min) <= 2.2) and (single_corr >= 0.66):
            orph_ok = True

    cliff_ok = bool((bsqi >= 0.80) and (kurt >= 5.0) and (bas_sqi >= 0.90) and (len(peaks_a) >= c.morph_min_beats))

    feat_vec = np.asarray([miss_bsqi, morph_disp, bas_noise, hf_noise, kurt_deficit], dtype=float)
    return {
        "features": feat_vec,
        "peaks_a_s": peaks_a,
        "peaks_b_s": peaks_b,
        "bsqi": bsqi,
        "morph_disp": morph_disp,
        "single_corr": single_corr,
        "bas_sqi": bas_sqi,
        "hf_noise": hf_noise,
        "kurt": kurt,
        "orphanidou_accept": orph_ok,
        "clifford_accept": cliff_ok,
    }
