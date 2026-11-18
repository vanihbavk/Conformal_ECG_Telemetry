"""Patient-Cluster Conformal Risk Control for Ambulatory ECG Quality Gating."""

from .afe import apply_diagnostic_afe_360hz, block_haar_swt_denoise, causal_clinical_bandpass_360hz
from .config import AfeConfig, ConformalConfig, MetricsConfig, SqiConfig
from .conformal import (
    calibrate_crc_threshold,
    calibrate_group_crc_thresholds,
    calibrate_iid_threshold,
    evaluate_gating_decision,
    fit_robust_score_params,
    patient_cluster_bootstrap_ci,
    score_uncertainty,
    score_unsupervised,
    simulate_clustered_correlation_sweep,
    wilson_binomial_ci,
)
from .datasets import (
    MITDB_RECORDS_48,
    PACED_RECORDS,
    generate_ambulatory_corpus,
    inject_stress_noise,
    load_mitdb_contiguous_streams,
    load_nstdb_360hz,
)
from .metrics import compute_qrs_morphology_prd, compute_rmssd_ms, evaluate_clinical_window, match_beats_aami
from .sqi import compute_bsqi, compute_two_template_morph_disp, detect_qrs_pantompkins, detect_qrs_steepness, extract_sqi_features

__all__ = [
    "AfeConfig",
    "ConformalConfig",
    "MetricsConfig",
    "MITDB_RECORDS_48",
    "PACED_RECORDS",
    "SqiConfig",
    "apply_diagnostic_afe_360hz",
    "block_haar_swt_denoise",
    "calibrate_crc_threshold",
    "calibrate_group_crc_thresholds",
    "calibrate_iid_threshold",
    "causal_clinical_bandpass_360hz",
    "compute_bsqi",
    "compute_qrs_morphology_prd",
    "compute_rmssd_ms",
    "compute_two_template_morph_disp",
    "detect_qrs_pantompkins",
    "detect_qrs_steepness",
    "evaluate_clinical_window",
    "evaluate_gating_decision",
    "extract_sqi_features",
    "fit_robust_score_params",
    "generate_ambulatory_corpus",
    "inject_stress_noise",
    "load_mitdb_contiguous_streams",
    "load_nstdb_360hz",
    "match_beats_aami",
    "patient_cluster_bootstrap_ci",
    "score_uncertainty",
    "score_unsupervised",
    "simulate_clustered_correlation_sweep",
    "wilson_binomial_ci",
]
