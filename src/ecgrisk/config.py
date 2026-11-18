"""Configuration dataclasses for ecgrisk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AfeConfig:
    """Analog front-end and filter parameters."""

    fs: float = 360.0
    diag_hp_hz: float = 0.5
    diag_lp_hz: float = 150.0
    diag_filter_order: int = 2
    adc_bits: int = 12
    adc_range_mv: float = 10.0
    clin_hp_hz: float = 0.5
    clin_lp_hz: float = 40.0
    clin_filter_order: int = 4
    wavelet_levels: int = 4
    wavelet_name: str = "haar"


@dataclass(frozen=True)
class SqiConfig:
    """Signal quality index and QRS detector parameters."""

    pt_bp_low: float = 5.0
    pt_bp_high: float = 15.0
    pt_mwi_s: float = 0.15
    pt_refractory_s: float = 0.20
    pt_min_dist_s: float = 0.25
    pt_thresh_pct: float = 45.0
    pt_thresh_min: float = 0.16
    steep_bp_low: float = 10.0
    steep_bp_high: float = 35.0
    steep_env_s: float = 0.045
    steep_min_dist_s: float = 0.22
    steep_thresh_pct: float = 25.0
    steep_thresh_min: float = 0.24
    bsqi_tol_s: float = 0.075
    morph_half_window_s: float = 0.09
    morph_cluster_corr_thresh: float = 0.78
    morph_secondary_coherence_min: float = 0.85
    morph_min_cluster_beats: int = 2
    morph_min_beats: int = 4
    morph_outlier_penalty: float = 0.85
    bas_cutoff_hz: float = 1.0
    hf_noise_low: float = 40.0
    hf_noise_high: float = 150.0
    hf_ref_low: float = 5.0
    hf_ref_high: float = 25.0
    kurt_target: float = 6.0
    kurt_scale: float = 6.0
    kurt_clip_max: float = 1.2


@dataclass(frozen=True)
class ConformalConfig:
    """Conformal risk control parameters."""

    z_clip_max: float = 6.0
    min_weight: float = 0.05
    fixed_unsup_scales: tuple[float, ...] = (0.25, 0.25, 0.25, 0.50, 0.50)
    fixed_unsup_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    target_alphas: tuple[float, ...] = (0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
    headline_alpha: float = 0.08


@dataclass(frozen=True)
class MetricsConfig:
    """Downstream clinical failure thresholds."""

    match_tolerance_s: float = 0.075
    min_beat_f1: float = 0.90
    max_rmssd_ms: float = 20.0
    max_rmssd_rel: float = 0.25
    max_qrs_prd: float = 25.0
