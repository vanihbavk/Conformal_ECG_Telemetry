"""Causal 360 Hz AFE simulation, clinical bandpass, and Haar wavelet denoising."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .config import AfeConfig

_DEFAULT_AFE = AfeConfig()


def apply_diagnostic_afe_360hz(
    sig_mv: np.ndarray,
    fs: float | None = None,
    adc_bits: int | None = None,
    v_range_mv: float | None = None,
    cfg: AfeConfig | None = None,
) -> np.ndarray:
    """Causal 0.5-150 Hz bandpass and 12-bit ADC quantization."""
    c = cfg or _DEFAULT_AFE
    fs_v = c.fs if fs is None else fs
    bits_v = c.adc_bits if adc_bits is None else adc_bits
    range_v = c.adc_range_mv if v_range_mv is None else v_range_mv

    hp_sos = butter(c.diag_filter_order, c.diag_hp_hz, btype="highpass", fs=fs_v, output="sos")
    lp_sos = butter(c.diag_filter_order, min(c.diag_lp_hz, 0.48 * fs_v), btype="lowpass", fs=fs_v, output="sos")
    x = sig_mv.astype(float) - float(np.median(sig_mv[: min(len(sig_mv), int(0.5 * fs_v))]))
    x = sosfilt(hp_sos, x)
    x = sosfilt(lp_sos, x)
    half_fsr = 0.5 * range_v
    lsb = range_v / float(2**bits_v)
    x = np.clip(x, -half_fsr, half_fsr - lsb)
    return np.round(x / lsb) * lsb


def causal_clinical_bandpass_360hz(
    sig_afe: np.ndarray,
    fs: float | None = None,
    lo_hz: float | None = None,
    hi_hz: float | None = None,
    cfg: AfeConfig | None = None,
) -> np.ndarray:
    """Causal 0.5-40 Hz Butterworth clinical monitor filter."""
    c = cfg or _DEFAULT_AFE
    fs_v = c.fs if fs is None else fs
    lo_v = c.clin_hp_hz if lo_hz is None else lo_hz
    hi_v = c.clin_lp_hz if hi_hz is None else hi_hz

    hp_sos = butter(c.diag_filter_order, lo_v, btype="highpass", fs=fs_v, output="sos")
    lp_sos = butter(c.clin_filter_order, min(hi_v, 0.45 * fs_v), btype="lowpass", fs=fs_v, output="sos")
    y = sosfilt(hp_sos, sig_afe)
    y = sosfilt(lp_sos, y)
    return y


def block_haar_swt_denoise(
    sig_afe: np.ndarray,
    fs: float | None = None,
    levels: int | None = None,
    cfg: AfeConfig | None = None,
) -> np.ndarray:
    """Undecimated Haar wavelet soft-thresholding after causal highpass."""
    c = cfg or _DEFAULT_AFE
    fs_v = c.fs if fs is None else fs
    lev_v = c.wavelet_levels if levels is None else levels

    hp_sos = butter(c.diag_filter_order, c.clin_hp_hz, btype="highpass", fs=fs_v, output="sos")
    x = sosfilt(hp_sos, sig_afe.astype(float))
    n = len(x)
    if n < 32:
        return x

    approx = x.copy()
    details: list[np.ndarray] = []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for lev in range(lev_v):
        step = 1 << lev
        shifted = np.roll(approx, step)
        shifted[:step] = approx[:step]
        a_next = inv_sqrt2 * (approx + shifted)
        d_lev = inv_sqrt2 * (approx - shifted)
        details.append(d_lev)
        approx = a_next

    sigma_hat = float(np.median(np.abs(details[0]))) / 0.6745
    thresh = sigma_hat * np.sqrt(2.0 * np.log(max(n, 2)))
    thresh_details: list[np.ndarray] = []
    for lev, d_lev in enumerate(details):
        if lev < 3:
            scale = thresh * (0.85 ** lev)
            d_thr = np.sign(d_lev) * np.maximum(np.abs(d_lev) - scale, 0.0)
            thresh_details.append(d_thr)
        else:
            thresh_details.append(d_lev)

    rec = approx
    for lev in range(lev_v - 1, -1, -1):
        step = 1 << lev
        d_lev = thresh_details[lev]
        r_plus = inv_sqrt2 * (rec + d_lev)
        r_minus = np.roll(inv_sqrt2 * (rec - d_lev), -step)
        r_minus[-step:] = r_plus[-step:]
        rec = 0.5 * (r_plus + r_minus)
    return rec
