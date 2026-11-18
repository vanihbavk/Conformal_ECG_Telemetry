"""MIT-BIH + NSTDB patient-cluster CRC benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LATEX_DIR = RESULTS_DIR / "latex"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = LATEX_DIR / "tables"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ecgrisk.afe import apply_diagnostic_afe_360hz, block_haar_swt_denoise, causal_clinical_bandpass_360hz
from ecgrisk.config import ConformalConfig
from ecgrisk.conformal import (
    calibrate_crc_threshold,
    calibrate_group_crc_thresholds,
    calibrate_iid_threshold,
    evaluate_endpoint_distributions,
    evaluate_gating_decision,
    fit_robust_score_params,
    patient_cluster_bootstrap_ci,
    patient_clustered_validity_test,
    score_uncertainty,
    score_unsupervised,
    simulate_clustered_correlation_sweep,
)
from ecgrisk.datasets import (
    MITDB_RECORDS_48,
    generate_ambulatory_corpus,
    inject_stress_noise,
    load_mitdb_contiguous_streams,
    load_nstdb_360hz,
)
from ecgrisk.metrics import evaluate_clinical_window
from ecgrisk.sqi import extract_sqi_features

_CRC_CFG = ConformalConfig()
ALPHAS: tuple[float, ...] = _CRC_CFG.target_alphas
FEAT_NAMES: tuple[str, ...] = ("1 - bSQI", "MorphDisp", "1 - basSQI", "HF Ratio", "Kurt Deficit")


def _process_corpus(corpus: list[dict[str, Any]], denoiser: str = "bandpass") -> dict[str, Any]:
    """Extract zero-oracle SQIs and clinical failure endpoints across all strips."""
    n = len(corpus)
    pids, groups, noise_regimes = [], [], []
    strip_idx = np.zeros(n, dtype=int)
    has_ectopic = np.zeros(n, dtype=bool)
    features = np.zeros((n, 5), dtype=float)
    single_corrs = np.zeros(n, dtype=float)
    orph_acc = np.zeros(n, dtype=bool)
    cliff_acc = np.zeros(n, dtype=bool)

    beat_f1 = np.zeros(n, dtype=float)
    ect_recall = np.full(n, np.nan, dtype=float)
    rmssd_err = np.zeros(n, dtype=float)
    qrs_prd = np.zeros(n, dtype=float)
    r_amp_err = np.zeros(n, dtype=float)
    losses_nom = np.zeros(n, dtype=float)
    losses_det_hrv = np.zeros(n, dtype=float)
    losses_strict = np.zeros(n, dtype=float)
    losses_lenient = np.zeros(n, dtype=float)

    for idx, row in enumerate(corpus):
        fs = float(row["fs"])
        afe_clean = apply_diagnostic_afe_360hz(row["sig_clean_mv"], fs=fs)
        afe_obs = apply_diagnostic_afe_360hz(row["sig_obs_mv"], fs=fs)
        if denoiser == "wavelet":
            clin_clean = block_haar_swt_denoise(afe_clean, fs=fs)
            clin_obs = block_haar_swt_denoise(afe_obs, fs=fs)
        else:
            clin_clean = causal_clinical_bandpass_360hz(afe_clean, fs=fs)
            clin_obs = causal_clinical_bandpass_360hz(afe_obs, fs=fs)

        sqi = extract_sqi_features(afe_obs, clin_obs, fs=fs)
        cm = evaluate_clinical_window(
            clin_obs,
            clin_clean,
            row["r_peaks_s"],
            sqi["peaks_a_s"],
            ect_flags=row["ect_flags"],
            fs=fs,
        )

        pids.append(row["pid"])
        groups.append(row["group"])
        strip_idx[idx] = int(row["strip_idx"])
        noise_regimes.append(row["noise_regime"])
        has_ectopic[idx] = bool(row["has_ectopic"])
        features[idx] = sqi["features"]
        single_corrs[idx] = float(sqi["single_corr"])
        orph_acc[idx] = bool(sqi["orphanidou_accept"])
        cliff_acc[idx] = bool(sqi["clifford_accept"])
        beat_f1[idx] = cm["beat_f1"]
        ect_recall[idx] = cm["ectopic_recall"]
        rmssd_err[idx] = cm["rmssd_err_ms"]
        qrs_prd[idx] = cm["qrs_prd_pct"]
        r_amp_err[idx] = cm["r_amp_err_pct"]
        losses_nom[idx] = cm["clin_fail"]
        losses_det_hrv[idx] = cm["clin_fail_det_hrv"]
        losses_strict[idx] = cm["clin_fail_strict"]
        losses_lenient[idx] = cm["clin_fail_lenient"]

    return {
        "pid": np.asarray(pids),
        "group": np.asarray(groups),
        "strip_idx": strip_idx,
        "noise_regime": np.asarray(noise_regimes),
        "is_clean": np.asarray([r == "clean" for r in noise_regimes], dtype=bool),
        "has_ectopic": has_ectopic,
        "features": features,
        "single_corr": single_corrs,
        "orph_accept": orph_acc,
        "cliff_accept": cliff_acc,
        "beat_f1": beat_f1,
        "ectopic_recall": ect_recall,
        "rmssd_err_ms": rmssd_err,
        "qrs_prd_pct": qrs_prd,
        "r_amp_err_pct": r_amp_err,
        "loss_nom": losses_nom,
        "loss_det_hrv": losses_det_hrv,
        "loss_strict": losses_strict,
        "loss_lenient": losses_lenient,
    }


def _patient_risks_for_mask(
    accept: np.ndarray,
    losses: np.ndarray,
    pids: np.ndarray,
    subset_mask: np.ndarray | None = None,
) -> list[float]:
    if subset_mask is None:
        subset_mask = np.ones(len(pids), dtype=bool)
    return [
        float(np.mean(np.where(accept[subset_mask & (pids == pid)], losses[subset_mask & (pids == pid)], 0.0)))
        for pid in sorted(np.unique(pids[subset_mask]))
    ]


def run_multi_seed_cv(
    datasets_by_noise: list[dict[str, Any]],
    loss_key: str = "loss_nom",
    splits_per_noise: int = 4,
    n_folds: int = 6,
    include_baselines: bool = True,
) -> dict[str, Any]:
    """Run patient-disjoint 6-fold CV across noise seeds and split permutations."""
    ref_data = datasets_by_noise[0]
    all_pids = np.asarray(sorted(np.unique(ref_data["pid"])))
    dummy_folds = np.array_split(all_pids, n_folds)
    m_test_min = int(min(len(f) for f in dummy_folds))
    m_test_max = int(max(len(f) for f in dummy_folds))
    min_train = int(min(len(all_pids) - len(f) for f in dummy_folds))
    n_fit = int(min(12, min_train // 2))
    n_cal_sup = int(min_train - n_fit)
    n_runs = len(datasets_by_noise) * splits_per_noise

    baseline_keys = (
        ("single_bsqi", "single_morph", "single_bas", "single_hf", "single_kurt", "heur_orph", "core_2tpl", "heur_cliff", "scale_unit", "ml_lr", "ml_gb")
        if include_baselines
        else ()
    )

    out: dict[str, Any] = {
        "n_runs": n_runs,
        "n_noise_seeds": len(datasets_by_noise),
        "splits_per_noise": splits_per_noise,
        "n_folds": n_folds,
        "n_patients": int(len(all_pids)),
        "m_fit": n_fit,
        "m_cal_sup": n_cal_sup,
        "m_cal": min_train,
        "m_train": min_train,
        "m_test_min": m_test_min,
        "m_test_max": m_test_max,
        "cal_slack_pct": 100.0 / (min_train + 1.0),
        "cal_slack_sup_pct": 100.0 / (n_cal_sup + 1.0),
        "alphas": list(ALPHAS),
        "crc": {a: [] for a in ALPHAS},
        "crc_m27": {a: [] for a in ALPHAS},
        "crc_sup": {a: [] for a in ALPHAS},
        "iid": {a: [] for a in ALPHAS},
        "iid_sup": {a: [] for a in ALPHAS},
        "causal_next": {a: [] for a in ALPHAS},
        **{bk: {a: [] for a in ALPHAS} for bk in baseline_keys},
        "crc_fold_violations": {a: 0 for a in ALPHAS},
        "iid_fold_violations": {a: 0 for a in ALPHAS},
        "tau_list": {a: [] for a in ALPHAS},
        "tau_sup_list": {a: [] for a in ALPHAS},
        "sup_meds": [],
        "sup_scales": [],
        "sup_weights": [],
        "seed0_accept_08": None,
        "seed0_accept_10": None,
        "seed0_folds_08": [],
        "seed0_distributions_08": {},
    }
    total_folds = n_runs * n_folds
    run_idx = 0

    for data in datasets_by_noise:
        losses = data[loss_key]
        features = data["features"]
        pids = data["pid"]
        n_win = len(pids)
        pid_to_idx = {p: np.where(pids == p)[0] for p in all_pids}

        scores_unsup = score_unsupervised(features)
        scores_unit = score_unsupervised(features, cfg=ConformalConfig(fixed_unsup_scales=(1.0, 1.0, 1.0, 1.0, 1.0)))
        cliff_cont = (features[:, 0] + features[:, 2] + features[:, 4]) / 3.0
        orph_cont = (features[:, 0] + np.clip(1.0 - data["single_corr"], 0.0, 1.0)) / 2.0
        core_2tpl = (features[:, 0] + features[:, 1]) / 2.0

        for _ in range(splits_per_noise):
            rng = np.random.default_rng(1000 + run_idx)
            perm_pids = all_pids.copy()
            rng.shuffle(perm_pids)
            folds = np.array_split(perm_pids, n_folds)

            oof_accept_crc = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_m27 = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_sup = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_iid = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_iid_sup = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_causal = {a: np.zeros(n_win, dtype=bool) for a in ALPHAS}
            oof_accept_base = {bk: {a: np.zeros(n_win, dtype=bool) for a in ALPHAS} for bk in baseline_keys}
            oof_causal_valid = np.zeros(n_win, dtype=bool)

            for f_idx in range(n_folds):
                test_pids = list(folds[f_idx])
                train_pids = [p for j, f in enumerate(folds) if j != f_idx for p in f]
                fit_pids = train_pids[:n_fit]
                cal_sup_pids = train_pids[n_fit : n_fit + n_cal_sup]

                fit_idx = np.concatenate([pid_to_idx[p] for p in fit_pids])
                cal_sup_idx = np.concatenate([pid_to_idx[p] for p in cal_sup_pids])
                train_all_idx = np.concatenate([pid_to_idx[p] for p in train_pids])
                test_idx = np.concatenate([pid_to_idx[p] for p in test_pids])

                train_losses = losses[train_all_idx]
                train_p = pids[train_all_idx]
                cal_sup_losses = losses[cal_sup_idx]
                cal_sup_p = pids[cal_sup_idx]
                test_losses = losses[test_idx]
                test_p = pids[test_idx]

                cal_u = scores_unsup[train_all_idx]
                cal_u_m27 = scores_unsup[cal_sup_idx]
                test_u = scores_unsup[test_idx]

                med, scale, w = fit_robust_score_params(features[fit_idx], losses[fit_idx])
                out["sup_meds"].append(med)
                out["sup_scales"].append(scale)
                out["sup_weights"].append(w)
                cal_s = score_uncertainty(features[cal_sup_idx], med, scale, w)
                test_s = score_uncertainty(features[test_idx], med, scale, w)

                base_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                if include_baselines:
                    lr = LogisticRegression(C=1.0, max_iter=300, random_state=42)
                    lr.fit(features[fit_idx], losses[fit_idx])
                    cal_sc_lr = lr.predict_proba(features[cal_sup_idx])[:, 1]
                    test_sc_lr = lr.predict_proba(features[test_idx])[:, 1]

                    gb = HistGradientBoostingClassifier(max_iter=50, min_samples_leaf=15, random_state=42)
                    gb.fit(features[fit_idx], losses[fit_idx])
                    cal_sc_gb = gb.predict_proba(features[cal_sup_idx])[:, 1]
                    test_sc_gb = gb.predict_proba(features[test_idx])[:, 1]

                    base_map = {
                        "single_bsqi": (features[train_all_idx, 0], features[test_idx, 0], train_losses, train_p),
                        "single_morph": (features[train_all_idx, 1], features[test_idx, 1], train_losses, train_p),
                        "single_bas": (features[train_all_idx, 2], features[test_idx, 2], train_losses, train_p),
                        "single_hf": (features[train_all_idx, 3], features[test_idx, 3], train_losses, train_p),
                        "single_kurt": (features[train_all_idx, 4], features[test_idx, 4], train_losses, train_p),
                        "heur_orph": (orph_cont[train_all_idx], orph_cont[test_idx], train_losses, train_p),
                        "core_2tpl": (core_2tpl[train_all_idx], core_2tpl[test_idx], train_losses, train_p),
                        "heur_cliff": (cliff_cont[train_all_idx], cliff_cont[test_idx], train_losses, train_p),
                        "scale_unit": (scores_unit[train_all_idx], scores_unit[test_idx], train_losses, train_p),
                        "ml_lr": (cal_sc_lr, test_sc_lr, cal_sup_losses, cal_sup_p),
                        "ml_gb": (cal_sc_gb, test_sc_gb, cal_sup_losses, cal_sup_p),
                    }

                for alpha in ALPHAS:
                    tau_crc = calibrate_crc_threshold(cal_u, train_losses, train_p, alpha=alpha)
                    tau_m27 = calibrate_crc_threshold(cal_u_m27, cal_sup_losses, cal_sup_p, alpha=alpha)
                    tau_sup = calibrate_crc_threshold(cal_s, cal_sup_losses, cal_sup_p, alpha=alpha)
                    tau_iid = calibrate_iid_threshold(cal_u, train_losses, alpha=alpha)
                    tau_iid_sup = calibrate_iid_threshold(cal_s, cal_sup_losses, alpha=alpha)

                    out["tau_list"][alpha].append(float(tau_crc))
                    out["tau_sup_list"][alpha].append(float(tau_sup))
                    if run_idx == 0 and abs(alpha - 0.08) < 1e-9:
                        out["seed0_folds_08"].append(
                            {"test_pids": list(test_pids), "tau": float(tau_crc), "tau_sup": float(tau_sup)}
                        )

                    acc_c = test_u <= tau_crc
                    oof_accept_crc[alpha][test_idx] = acc_c
                    oof_accept_m27[alpha][test_idx] = test_u <= tau_m27
                    oof_accept_sup[alpha][test_idx] = test_s <= tau_sup
                    oof_accept_iid[alpha][test_idx] = test_u <= tau_iid
                    oof_accept_iid_sup[alpha][test_idx] = test_s <= tau_iid_sup

                    for bk, (b_cal, b_test, b_loss, b_pids) in base_map.items():
                        tau_b = calibrate_crc_threshold(b_cal, b_loss, b_pids, alpha=alpha)
                        oof_accept_base[bk][alpha][test_idx] = b_test <= tau_b

                    fold_risk_c = float(np.mean(_patient_risks_for_mask(acc_c, test_losses, test_p)))
                    fold_risk_i = float(np.mean(_patient_risks_for_mask(test_u <= tau_iid, test_losses, test_p)))
                    if fold_risk_c > alpha + 1e-12:
                        out["crc_fold_violations"][alpha] += 1
                    if fold_risk_i > alpha + 1e-12:
                        out["iid_fold_violations"][alpha] += 1

                    for tp in test_pids:
                        p_rows = pid_to_idx[tp]
                        p_rows = p_rows[np.argsort(data["strip_idx"][p_rows])]
                        p_scores = scores_unsup[p_rows]
                        for k in range(1, len(p_rows)):
                            idx_next = p_rows[k]
                            oof_causal_valid[idx_next] = True
                            oof_accept_causal[alpha][idx_next] = bool(p_scores[k - 1] <= tau_crc)

            if run_idx == 0:
                out["seed0_accept_08"] = oof_accept_crc[0.08].copy()
                out["seed0_accept_10"] = oof_accept_crc[0.10].copy()
                m_c08 = oof_accept_crc[0.08]
                m_i08 = oof_accept_iid[0.08]
                out["seed0_distributions_08"] = {
                    "crc": {
                        "beat_f1": evaluate_endpoint_distributions(data["beat_f1"][m_c08]),
                        "qrs_prd": evaluate_endpoint_distributions(data["qrs_prd_pct"][m_c08]),
                        "rmssd_err": evaluate_endpoint_distributions(data["rmssd_err_ms"][m_c08]),
                    },
                    "iid": {
                        "beat_f1": evaluate_endpoint_distributions(data["beat_f1"][m_i08]),
                        "qrs_prd": evaluate_endpoint_distributions(data["qrs_prd_pct"][m_i08]),
                        "rmssd_err": evaluate_endpoint_distributions(data["rmssd_err_ms"][m_i08]),
                    },
                }

            eval_tuples = [
                ("crc", oof_accept_crc, None),
                ("crc_m27", oof_accept_m27, None),
                ("crc_sup", oof_accept_sup, None),
                ("iid", oof_accept_iid, None),
                ("iid_sup", oof_accept_iid_sup, None),
                ("causal_next", oof_accept_causal, oof_causal_valid),
            ] + [(bk, oof_accept_base[bk], None) for bk in baseline_keys]

            for alpha in ALPHAS:
                for key, acc_dict, mask in eval_tuples:
                    acc_arr = acc_dict[alpha]
                    eval_mask = np.ones(n_win, dtype=bool) if mask is None else mask
                    metrics = evaluate_gating_decision(
                        acc_arr[eval_mask],
                        losses[eval_mask],
                        data["beat_f1"][eval_mask],
                        data["rmssd_err_ms"][eval_mask],
                        data["qrs_prd_pct"][eval_mask],
                        data["ectopic_recall"][eval_mask],
                    )
                    pat_r = _patient_risks_for_mask(acc_arr, losses, pids, subset_mask=eval_mask)
                    metrics["patient_risk_mean"] = float(np.mean(pat_r))
                    metrics["patient_risk_values"] = pat_r
                    out[key][alpha].append(metrics)

            run_idx += 1

    out["total_folds"] = total_folds
    out["sup_medians_mean"] = np.mean(np.asarray(out.pop("sup_meds")), axis=0).tolist()
    out["sup_scales_mean"] = np.mean(np.asarray(out.pop("sup_scales")), axis=0).tolist()
    out["sup_weights_mean"] = np.mean(np.asarray(out.pop("sup_weights")), axis=0).tolist()
    return out


def _aggregate_seed_list(seed_list: list[dict[str, Any]], alpha: float = 0.08) -> dict[str, Any]:
    """Summarize multi-run evaluation with cluster-bootstrap 95% CI and clustered t-test."""
    pat_matrix = np.asarray([s["patient_risk_values"] for s in seed_list], dtype=float)
    per_pat_mean = np.mean(pat_matrix, axis=0).tolist()
    risk_mean, risk_ci_lo, risk_ci_hi = patient_cluster_bootstrap_ci(per_pat_mean, n_boot=2000, seed=42)
    val_test = patient_clustered_validity_test(per_pat_mean, alpha=alpha)

    seed_risks = np.asarray([s["patient_risk_mean"] for s in seed_list], dtype=float) * 100.0
    accs = np.asarray([s["accept_pct"] for s in seed_list], dtype=float)
    absts = np.asarray([s["abstain_pct"] for s in seed_list], dtype=float)
    conds = np.asarray([s["cond_fail_pct"] for s in seed_list], dtype=float)
    f1s = np.asarray([s["acc_beat_f1_med"] for s in seed_list], dtype=float)
    rmssds = np.asarray([s["acc_rmssd_err_med"] for s in seed_list], dtype=float)
    prds = np.asarray([s["acc_qrs_prd_med"] for s in seed_list], dtype=float)
    ects = np.asarray([s["acc_ect_recall_pct"] for s in seed_list], dtype=float)
    ect_wins = np.asarray([s["ect_win_accept_pct"] for s in seed_list], dtype=float)

    return {
        "patient_risk_pct": risk_mean * 100.0,
        "patient_risk_ci_lo_pct": risk_ci_lo * 100.0,
        "patient_risk_ci_hi_pct": risk_ci_hi * 100.0,
        "patient_risk_seed_sd_pct": float(np.std(seed_risks)),
        "accept_pct": float(np.mean(accs)),
        "accept_sd_pct": float(np.std(accs)),
        "abstain_pct": float(np.mean(absts)),
        "cond_fail_pct": float(np.mean(conds)),
        "p_viol": val_test["p_viol"],
        "t_stat": val_test["t_stat"],
        "stud_ci_lo_pct": val_test["ci_lo"] * 100.0,
        "stud_ci_hi_pct": val_test["ci_hi"] * 100.0,
        "acc_beat_f1_med": float(np.median(f1s[np.isfinite(f1s)])) if np.any(np.isfinite(f1s)) else np.nan,
        "acc_rmssd_err_med": float(np.median(rmssds[np.isfinite(rmssds)])) if np.any(np.isfinite(rmssds)) else np.nan,
        "acc_qrs_prd_med": float(np.median(prds[np.isfinite(prds)])) if np.any(np.isfinite(prds)) else np.nan,
        "acc_ect_recall_pct": float(np.mean(ects[np.isfinite(ects)])) if np.any(np.isfinite(ects)) else np.nan,
        "ect_win_accept_pct": float(np.mean(ect_wins[np.isfinite(ect_wins)])) if np.any(np.isfinite(ect_wins)) else np.nan,
    }


def _evaluate_fixed_rule(data: dict[str, Any], acc_mask: np.ndarray, loss_key: str = "loss_nom") -> dict[str, float]:
    losses = data[loss_key]
    m = evaluate_gating_decision(
        acc_mask,
        losses,
        data["beat_f1"],
        data["rmssd_err_ms"],
        data["qrs_prd_pct"],
        data["ectopic_recall"],
    )
    pat_r = _patient_risks_for_mask(acc_mask, losses, data["pid"])
    rm, rlo, rhi = patient_cluster_bootstrap_ci(pat_r, n_boot=2000, seed=42)
    m["patient_risk_pct"] = rm * 100.0
    m["patient_risk_ci_lo_pct"] = rlo * 100.0
    m["patient_risk_ci_hi_pct"] = rhi * 100.0
    return m


def run_subgroup_mondrian_study(
    datasets_by_noise: list[dict[str, Any]],
    n_folds: int = 6,
) -> dict[str, Any]:
    """Compare marginal vs Mondrian group-conditional CRC across Sinus, Ectopic, and Paced."""
    ref_data = datasets_by_noise[0]
    all_pids = np.asarray(sorted(np.unique(ref_data["pid"])))
    pid_group = {pid: str(ref_data["group"][ref_data["pid"] == pid][0]) for pid in all_pids}

    out: dict[str, Any] = {}
    for target_alpha in (0.08, 0.10):
        marg_risks: dict[str, list[float]] = {"Sinus": [], "Ectopic": [], "Paced": []}
        marg_acc: dict[str, list[float]] = {"Sinus": [], "Ectopic": [], "Paced": []}
        mond_risks: dict[str, list[float]] = {"Sinus": [], "Ectopic": [], "Paced": []}
        mond_acc: dict[str, list[float]] = {"Sinus": [], "Ectopic": [], "Paced": []}
        feas_counts = {"Sinus": 0, "Ectopic": 0, "Paced": 0}
        total_folds = 0

        for n_idx, data in enumerate(datasets_by_noise):
            scores = score_unsupervised(data["features"])
            losses = data["loss_nom"]
            pids = data["pid"]
            groups = data["group"]
            pid_to_idx = {p: np.where(pids == p)[0] for p in all_pids}

            rng = np.random.default_rng(3000 + n_idx)
            perm_pids = all_pids.copy()
            rng.shuffle(perm_pids)
            folds = np.array_split(perm_pids, n_folds)

            oof_marg = np.zeros(len(pids), dtype=bool)
            oof_mond = np.zeros(len(pids), dtype=bool)

            for f_idx in range(n_folds):
                total_folds += 1
                test_pids = list(folds[f_idx])
                train_pids = [p for j, f in enumerate(folds) if j != f_idx for p in f]
                train_idx = np.concatenate([pid_to_idx[p] for p in train_pids])
                test_idx = np.concatenate([pid_to_idx[p] for p in test_pids])

                tau_marg = calibrate_crc_threshold(scores[train_idx], losses[train_idx], pids[train_idx], alpha=target_alpha)
                oof_marg[test_idx] = scores[test_idx] <= tau_marg

                g_cal_info = calibrate_group_crc_thresholds(
                    scores[train_idx], losses[train_idx], pids[train_idx], groups[train_idx], alpha=target_alpha
                )
                for g_name in ("Sinus", "Ectopic", "Paced"):
                    if g_name in g_cal_info and g_cal_info[g_name]["feasible"]:
                        feas_counts[g_name] += 1

                for idx in test_idx:
                    g = str(groups[idx])
                    tau_g = float(g_cal_info[g]["tau"]) if g in g_cal_info else -np.inf
                    oof_mond[idx] = bool(scores[idx] <= tau_g)

            for g_name in ("Sinus", "Ectopic", "Paced"):
                g_mask = groups == g_name
                marg_acc[g_name].append(float(np.mean(oof_marg[g_mask])) * 100.0)
                mond_acc[g_name].append(float(np.mean(oof_mond[g_mask])) * 100.0)
                marg_risks[g_name].append(float(np.mean(_patient_risks_for_mask(oof_marg, losses, pids, g_mask))) * 100.0)
                mond_risks[g_name].append(float(np.mean(_patient_risks_for_mask(oof_mond, losses, pids, g_mask))) * 100.0)

        out[f"{target_alpha:.2f}"] = {
            g: {
                "n_patients": sum(1 for p in all_pids if pid_group[p] == g),
                "feasible_fold_pct": float(feas_counts[g]) / float(total_folds) * 100.0,
                "marginal_accept_pct": float(np.mean(marg_acc[g])),
                "marginal_risk_pct": float(np.mean(marg_risks[g])),
                "mondrian_accept_pct": float(np.mean(mond_acc[g])),
                "mondrian_risk_pct": float(np.mean(mond_risks[g])),
            }
            for g in ("Sinus", "Ectopic", "Paced")
        }
    return out


def run_conditional_ltt_study(
    data: dict[str, Any],
    n_seeds: int = 10,
    n_folds: int = 6,
    alpha_cond_targets: list[float] | None = None,
) -> dict[str, Any]:
    """Learn-then-Test (LTT) selective risk control on conditional failure rate."""
    if alpha_cond_targets is None:
        alpha_cond_targets = [0.05, 0.08, 0.10, 0.12, 0.15]

    scores = score_unsupervised(data["features"])
    losses = data["loss_nom"]
    pids = data["pid"]
    all_pids = np.asarray(sorted(np.unique(pids)))
    pid_to_idx = {p: np.where(pids == p)[0] for p in all_pids}

    delta = 0.10
    results = {f"{ac:.2f}": {"test_cond_risks": [], "test_accept_pct": []} for ac in alpha_cond_targets}

    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        perm_pids = all_pids.copy()
        rng.shuffle(perm_pids)
        folds = np.array_split(perm_pids, n_folds)

        for f_idx in range(n_folds):
            test_pids = list(folds[f_idx])
            train_pids = [p for j, f in enumerate(folds) if j != f_idx for p in f]
            cal_idx = np.concatenate([pid_to_idx[p] for p in train_pids])
            test_idx = np.concatenate([pid_to_idx[p] for p in test_pids])

            cal_losses = losses[cal_idx]
            cal_p = pids[cal_idx]
            test_losses = losses[test_idx]
            cal_sc = scores[cal_idx]
            test_sc = scores[test_idx]

            cand_taus = np.quantile(cal_sc, np.linspace(0.05, 0.95, 30))
            unique_cal_p = np.unique(cal_p)

            for ac in alpha_cond_targets:
                k = f"{ac:.2f}"
                best_tau = -np.inf
                for tau in sorted(cand_taus):
                    cal_acc = cal_sc <= tau
                    if np.sum(cal_acc) < 10:
                        continue
                    pat_cond_risks = []
                    for cp in unique_cal_p:
                        c_mask = (cal_p == cp) & cal_acc
                        if np.sum(c_mask) > 0:
                            pat_cond_risks.append(float(np.mean(cal_losses[c_mask])))
                    if len(pat_cond_risks) < 5:
                        continue
                    m_cr = np.mean(pat_cond_risks)
                    s_cr = np.std(pat_cond_risks, ddof=1) if len(pat_cond_risks) > 1 else 0.5
                    t_val = stats.t.ppf(1.0 - (delta / len(cand_taus)), df=max(len(pat_cond_risks) - 1, 1))
                    if m_cr + t_val * s_cr / np.sqrt(len(pat_cond_risks)) <= ac:
                        best_tau = tau

                test_acc = np.zeros(len(test_idx), dtype=bool) if best_tau == -np.inf else (test_sc <= best_tau)
                if np.any(test_acc):
                    t_cond_risk = float(np.mean(test_losses[test_acc])) * 100.0
                    t_acc_pct = float(np.mean(test_acc)) * 100.0
                else:
                    t_cond_risk = 0.0
                    t_acc_pct = 0.0

                results[k]["test_cond_risks"].append(t_cond_risk)
                results[k]["test_accept_pct"].append(t_acc_pct)

    return {
        f"{ac:.2f}": {
            "target_cond_risk_pct": ac * 100.0,
            "empirical_cond_risk_pct": float(np.mean(results[f"{ac:.2f}"]["test_cond_risks"])),
            "accept_pct": float(np.mean(results[f"{ac:.2f}"]["test_accept_pct"])),
        }
        for ac in alpha_cond_targets
    }


def _write_tables_and_macros(summary: dict[str, Any]) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    cv = summary["cv_nominal"]
    heur = summary["heuristics"]
    wave = summary["cv_wavelet"]
    sens = summary["sensitivity"]
    stress = summary["stress_nstdb"]
    csub = summary["cohort_subgroups"]
    mond = summary["subgroup_mondrian"]
    corr = summary["correlation_sweep"]

    test_split_str = str(summary["m_test_min"]) if summary["m_test_min"] == summary["m_test_max"] else f"{summary['m_test_min']}\\text{{--}}{summary['m_test_max']}"
    lines1 = [
        r"\begin{table}[t]",
        r"\centering",
        (
            f"\\caption{{Held-out multi-seed comparison across {summary['n_patients']} MIT-BIH patients "
            f"({summary['n_runs']} runs: {summary['n_noise_seeds']} NSTDB noise seeds $\\times$ {summary['splits_per_noise']} 6-fold patient splits). "
            r"Primary Patient-CRC uses unsupervised fixed-scale scoring ($M{=}" + f"{summary['m_cal']}" + r"$ cal.\ patients); $M{=}" + f"{summary['m_cal_sup']}" + r"$ rows isolate the calibration sample-size and supervised-fit effects ($" + f"{summary['m_fit']}/{summary['m_cal_sup']}/{test_split_str}" + r"$).}"
        ),
        r"\label{tab:main_results}",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Method / Target $\alpha$ & Acc.\ (\%) & Abst.\ (\%) & Silent Risk [95\% CI] (\%) & $p_{\mathrm{val}}$ ($H_0$) & Ect.\ Acc.\ (\%) & Cond.\ Fail (\%) \\",
        r"\midrule",
        (
            f"All-Accept (No Gate) & 100.0 & 0.0 & "
            f"{heur['all_accept']['patient_risk_pct']:.2f} [{heur['all_accept']['patient_risk_ci_lo_pct']:.2f}, {heur['all_accept']['patient_risk_ci_hi_pct']:.2f}] & "
            f"--- & 100.0 & {heur['all_accept']['cond_fail_pct']:.2f} \\\\"
        ),
        (
            f"Orphanidou (2015) \\cite{{orphanidou2015signal}} & {heur['orphanidou']['accept_pct']:.1f} & {heur['orphanidou']['abstain_pct']:.1f} & "
            f"{heur['orphanidou']['patient_risk_pct']:.2f} [{heur['orphanidou']['patient_risk_ci_lo_pct']:.2f}, {heur['orphanidou']['patient_risk_ci_hi_pct']:.2f}] & "
            f"--- & {heur['orphanidou']['ect_win_accept_pct']:.1f} & {heur['orphanidou']['cond_fail_pct']:.2f} \\\\"
        ),
        (
            f"Clifford (2012) \\cite{{clifford2012signal}} & {heur['clifford']['accept_pct']:.1f} & {heur['clifford']['abstain_pct']:.1f} & "
            f"{heur['clifford']['patient_risk_pct']:.2f} [{heur['clifford']['patient_risk_ci_lo_pct']:.2f}, {heur['clifford']['patient_risk_ci_hi_pct']:.2f}] & "
            f"--- & {heur['clifford']['ect_win_accept_pct']:.1f} & {heur['clifford']['cond_fail_pct']:.2f} \\\\"
        ),
        r"\midrule",
    ]
    for a in (0.05, 0.08, 0.10):
        k = f"{a:.2f}"
        ri = cv["iid"][k]
        lines1.append(
            f"Window-i.i.d. ($\\alpha={a:.2f}$) & {ri['accept_pct']:.1f} & {ri['abstain_pct']:.1f} & "
            f"{ri['patient_risk_pct']:.2f} [{ri['patient_risk_ci_lo_pct']:.2f}, {ri['patient_risk_ci_hi_pct']:.2f}] & "
            f"{ri['p_viol']:.3f} & {ri['ect_win_accept_pct']:.1f} & {ri['cond_fail_pct']:.2f} \\\\"
        )
    lines1.append(r"\midrule")
    for a in (0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        k = f"{a:.2f}"
        rc = cv["crc"][k]
        lines1.append(
            f"Patient-CRC ($M{{=}}{summary['m_cal']}$, $\\alpha={a:.2f}$) & {rc['accept_pct']:.1f} & {rc['abstain_pct']:.1f} & "
            f"{rc['patient_risk_pct']:.2f} [{rc['patient_risk_ci_lo_pct']:.2f}, {rc['patient_risk_ci_hi_pct']:.2f}] & "
            f"{rc['p_viol']:.3f} & {rc['ect_win_accept_pct']:.1f} & {rc['cond_fail_pct']:.2f} \\\\"
        )
    r_m27 = cv["crc_m27"]["0.08"]
    r_sup = cv["crc_sup"]["0.08"]
    r_cau = cv["causal_next"]["0.08"]
    lines1.extend(
        [
            r"\midrule",
            (
                f"Patient-CRC Same-$S_0$ ($M{{=}}{summary['m_cal_sup']}$, $\\alpha{{=}}0.08$) & {r_m27['accept_pct']:.1f} & {r_m27['abstain_pct']:.1f} & "
                f"{r_m27['patient_risk_pct']:.2f} [{r_m27['patient_risk_ci_lo_pct']:.2f}, {r_m27['patient_risk_ci_hi_pct']:.2f}] & "
                f"{r_m27['p_viol']:.3f} & {r_m27['ect_win_accept_pct']:.1f} & {r_m27['cond_fail_pct']:.2f} \\\\"
            ),
            (
                f"Supervised-Wt CRC ($M{{=}}{summary['m_cal_sup']}$, $\\alpha{{=}}0.08$) & {r_sup['accept_pct']:.1f} & {r_sup['abstain_pct']:.1f} & "
                f"{r_sup['patient_risk_pct']:.2f} [{r_sup['patient_risk_ci_lo_pct']:.2f}, {r_sup['patient_risk_ci_hi_pct']:.2f}] & "
                f"{r_sup['p_viol']:.3f} & {r_sup['ect_win_accept_pct']:.1f} & {r_sup['cond_fail_pct']:.2f} \\\\"
            ),
            (
                f"Proactive Causal $w{{\\to}}w{{+}}1$ ($\\alpha{{=}}0.08$) & {r_cau['accept_pct']:.1f} & {r_cau['abstain_pct']:.1f} & "
                f"{r_cau['patient_risk_pct']:.2f} [{r_cau['patient_risk_ci_lo_pct']:.2f}, {r_cau['patient_risk_ci_hi_pct']:.2f}] & "
                f"{r_cau['p_viol']:.3f} & {r_cau['ect_win_accept_pct']:.1f} & {r_cau['cond_fail_pct']:.2f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (TAB_DIR / "tab_main.tex").write_text("\n".join(lines1) + "\n", encoding="utf-8")

    def _fmt_f1(val: float) -> str:
        return "---" if not np.isfinite(val) else f"{val:.3f}"

    lines_b = [
        r"\begin{table}[t]",
        r"\centering",
        (
            f"\\caption{{Patient-Cluster CRC as a model-agnostic risk wrapper ($\\alpha = 0.08$, {summary['n_patients']} MIT-BIH patients, "
            f"{summary['n_runs']} runs). Zero-training scores calibrate on $M={summary['m_cal']}$ patients; $M={summary['m_cal_sup']}$ rows evaluate split-sample cost ($M_{{\\text{{fit}}}}={summary['m_fit']}$).}}"
        ),
        r"\label{tab:baselines}",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Method / Score & Gate Arch.\ ($M_{\text{cal}}$) & Acc.\ (\%) & Silent Risk [95\% CI] (\%) & Cond.\ Fail (\%) & Ect.\ Acc.\ (\%) & Med.\ F1 \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Literature Heuristics (Fixed Operating Points, Uncalibrated)}} \\",
        (
            f"Orphanidou (2015) & Single-template rule (---) & {heur['orphanidou']['accept_pct']:.1f} & "
            f"{heur['orphanidou']['patient_risk_pct']:.2f} [{heur['orphanidou']['patient_risk_ci_lo_pct']:.2f}, {heur['orphanidou']['patient_risk_ci_hi_pct']:.2f}] & "
            f"{heur['orphanidou']['cond_fail_pct']:.2f} & {heur['orphanidou']['ect_win_accept_pct']:.1f} & {_fmt_f1(heur['orphanidou']['acc_beat_f1_med'])} \\\\"
        ),
        (
            f"Clifford (2012) & Multi-SQI rule (---) & {heur['clifford']['accept_pct']:.1f} & "
            f"{heur['clifford']['patient_risk_pct']:.2f} [{heur['clifford']['patient_risk_ci_lo_pct']:.2f}, {heur['clifford']['patient_risk_ci_hi_pct']:.2f}] & "
            f"{heur['clifford']['cond_fail_pct']:.2f} & {heur['clifford']['ect_win_accept_pct']:.1f} & {_fmt_f1(heur['clifford']['acc_beat_f1_med'])} \\\\"
        ),
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Single-Feature Gates Wrapped in Patient-Cluster CRC ($\alpha=0.08$, $M_{\text{cal}}=39$)}} \\",
    ]
    single_specs = [
        ("single_bsqi", r"Single $1 - \mathrm{bSQI}$", "Dual-detector consensus (39)"),
        ("single_morph", r"Single $\mathrm{MorphDisp}$", "Two-template dispersion (39)"),
        ("single_bas", r"Single $1 - \mathrm{basSQI}$", "Baseline wander power (39)"),
        ("single_hf", r"Single $\mathrm{HFRatio}$", "High-frequency EMG (39)"),
        ("single_kurt", r"Single $\mathrm{KurtDef}$", "Excess kurtosis deficit (39)"),
    ]
    for skey, slabel, sdesc in single_specs:
        srow = cv[skey]["0.08"]
        lines_b.append(
            f"{slabel} & {sdesc} & {srow['accept_pct']:.1f} & "
            f"{srow['patient_risk_pct']:.2f} [{srow['patient_risk_ci_lo_pct']:.2f}, {srow['patient_risk_ci_hi_pct']:.2f}] & "
            f"{srow['cond_fail_pct']:.2f} & {srow['ect_win_accept_pct']:.1f} & {_fmt_f1(srow['acc_beat_f1_med'])} \\\\"
        )
    lines_b.extend([
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Zero-Training Multi-Feature Gates Wrapped in Patient-Cluster CRC ($\alpha=0.08$, $M_{\text{cal}}=39$)}} \\",
    ])
    for hkey, hlabel, hdesc in [
        ("heur_orph", "Orphanidou (CRC)", r"1-Tpl $(f_1 + 1 - s_{\text{1-tpl}})/2$ (39)"),
        ("core_2tpl", "2-Tpl Core (CRC)", r"2-Tpl $(f_1 + f_2)/2$ (39)"),
        ("heur_cliff", "Clifford (CRC)", r"3-SQI $(f_1 + f_3 + f_5)/3$ (39)"),
        ("crc", r"\textbf{Unsup.\ Composite $S_0$ (CRC)}", r"5-SQI domain-scaled (39)"),
    ]:
        hrow = cv[hkey]["0.08"]
        lines_b.append(
            f"{hlabel} & {hdesc} & {hrow['accept_pct']:.1f} & "
            f"{hrow['patient_risk_pct']:.2f} [{hrow['patient_risk_ci_lo_pct']:.2f}, {hrow['patient_risk_ci_hi_pct']:.2f}] & "
            f"{hrow['cond_fail_pct']:.2f} & {hrow['ect_win_accept_pct']:.1f} & {_fmt_f1(hrow['acc_beat_f1_med'])} \\\\"
        )
    lines_b.extend([
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Reduced-Calibration ($M_{\text{cal}}=27$) and Supervised Fit-Fold Gates ($\alpha=0.08$, $M_{\text{fit}}=12$)}} \\",
    ])
    for mkey, mlabel, mdesc in [
        ("crc_m27", r"Unsup.\ Composite $S_0$ ($M{=}27$)", r"Same $S_0$, 27 cal patients (27)"),
        ("crc_sup", "Supervised-Wt Composite", r"Fit med/IQR/weights (27)"),
        ("ml_lr", "Logistic Regression", r"L2 regularized ($C=1.0$, 27)"),
        ("ml_gb", "HistGradientBoosting", r"Shallow trees ($d=3$, 27)"),
    ]:
        mrow = cv[mkey]["0.08"]
        lines_b.append(
            f"{mlabel} & {mdesc} & {mrow['accept_pct']:.1f} & "
            f"{mrow['patient_risk_pct']:.2f} [{mrow['patient_risk_ci_lo_pct']:.2f}, {mrow['patient_risk_ci_hi_pct']:.2f}] & "
            f"{mrow['cond_fail_pct']:.2f} & {mrow['ect_win_accept_pct']:.1f} & {_fmt_f1(mrow['acc_beat_f1_med'])} \\\\"
        )
    lines_b.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TAB_DIR / "tab_baselines.tex").write_text("\n".join(lines_b) + "\n", encoding="utf-8")

    w08 = wave["crc"]["0.08"]
    dh08 = sens["det_hrv"]["crc"]["0.08"]
    s08 = sens["strict"]["crc"]["0.08"]
    l08 = sens["lenient"]["crc"]["0.08"]
    u08 = cv["scale_unit"]["0.08"]
    n_tot_str = f"{summary['n_windows']:,}"
    lines2 = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Ablation across wavelet denoising, unit scaling, failure endpoints (including PRD-free Beat~F1+HRV, $\\alpha=0.08$), and zero-recalibration NSTDB stress shifts ($n={n_tot_str}$).}}",
        r"\label{tab:ablation_nstdb}",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Evaluation Regime / Pipeline & Accept (\%) & Silent Risk [95\% CI] (\%) & Med.\ Beat F1 & Med.\ QRS PRD (\%) \\",
        r"\midrule",
        (
            f"Nominal 0.5--40\\,Hz ($\\alpha=0.08$) & {cv['crc']['0.08']['accept_pct']:.1f} & "
            f"{cv['crc']['0.08']['patient_risk_pct']:.2f} [{cv['crc']['0.08']['patient_risk_ci_lo_pct']:.2f}, {cv['crc']['0.08']['patient_risk_ci_hi_pct']:.2f}] & "
            f"{cv['crc']['0.08']['acc_beat_f1_med']:.3f} & {cv['crc']['0.08']['acc_qrs_prd_med']:.1f} \\\\"
        ),
        (
            f"Unscaled Equal-Weight ($\\mathbf{{s}}=\\mathbf{{1}}$, $\\alpha=0.08$) & {u08['accept_pct']:.1f} & "
            f"{u08['patient_risk_pct']:.2f} [{u08['patient_risk_ci_lo_pct']:.2f}, {u08['patient_risk_ci_hi_pct']:.2f}] & "
            f"{u08['acc_beat_f1_med']:.3f} & {u08['acc_qrs_prd_med']:.1f} \\\\"
        ),
        (
            f"Haar SWT Wavelet ($\\alpha=0.08$) & {w08['accept_pct']:.1f} & "
            f"{w08['patient_risk_pct']:.2f} [{w08['patient_risk_ci_lo_pct']:.2f}, {w08['patient_risk_ci_hi_pct']:.2f}] & "
            f"{w08['acc_beat_f1_med']:.3f} & {w08['acc_qrs_prd_med']:.1f} \\\\"
        ),
        (
            f"PRD-Free Endpoint (Beat~F1 or HRV) & {dh08['accept_pct']:.1f} & "
            f"{dh08['patient_risk_pct']:.2f} [{dh08['patient_risk_ci_lo_pct']:.2f}, {dh08['patient_risk_ci_hi_pct']:.2f}] & "
            f"{dh08['acc_beat_f1_med']:.3f} & {dh08['acc_qrs_prd_med']:.1f} \\\\"
        ),
        (
            f"Strict Cutoff ($\\text{{F1}}<0.95$, $\\text{{PRD}}>15\\%$) & {s08['accept_pct']:.1f} & "
            f"{s08['patient_risk_pct']:.2f} [{s08['patient_risk_ci_lo_pct']:.2f}, {s08['patient_risk_ci_hi_pct']:.2f}] & "
            f"{s08['acc_beat_f1_med']:.3f} & {s08['acc_qrs_prd_med']:.1f} \\\\"
        ),
        (
            f"Lenient Cutoff ($\\text{{F1}}<0.85$, $\\text{{PRD}}>35\\%$) & {l08['accept_pct']:.1f} & "
            f"{l08['patient_risk_pct']:.2f} [{l08['patient_risk_ci_lo_pct']:.2f}, {l08['patient_risk_ci_hi_pct']:.2f}] & "
            f"{l08['acc_beat_f1_med']:.3f} & {l08['acc_qrs_prd_med']:.1f} \\\\"
        ),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Zero-Recalibration Disjoint Noise Stress (Interval 2: $[325\,\mathrm{k}, 650\,\mathrm{k}]$)}} \\",
    ]
    for sr in stress:
        lines2.append(
            f"{sr['tex_label']} (CRC 0.08) & {sr['crc08_accept_pct']:.1f} & "
            f"{sr['crc08_risk_pct']:.2f} [{sr['crc08_risk_ci_lo_pct']:.2f}, {sr['crc08_risk_ci_hi_pct']:.2f}] & "
            f"{sr['crc08_beat_f1_med']:.3f} & {sr['crc08_qrs_prd_med']:.1f} \\\\"
        )
        if "em" in sr["label"] and ("0dB" in sr["label"] or "-3dB" in sr["label"]):
            lines2.append(
                f"\\quad \\textit{{vs.}} Clifford et al. (2012) & {sr['cliff_accept_pct']:.1f} & "
                f"{sr['cliff_risk_pct']:.2f} [{sr['cliff_risk_ci_lo_pct']:.2f}, {sr['cliff_risk_ci_hi_pct']:.2f}] & --- & --- \\\\"
            )
            lines2.append(
                f"\\quad \\textit{{vs.}} Orphanidou et al. (2015) & {sr['orph_accept_pct']:.1f} & "
                f"{sr['orph_risk_pct']:.2f} [{sr['orph_risk_ci_lo_pct']:.2f}, {sr['orph_risk_ci_hi_pct']:.2f}] & --- & --- \\\\"
            )
    lines2.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TAB_DIR / "tab_clinical_nstdb.tex").write_text("\n".join(lines2) + "\n", encoding="utf-8")

    m = {
        "nPatients": f"{summary['n_patients']}",
        "nWindowsTotal": f"{summary['n_windows']:,}",
        "nRunsTotal": f"{summary['n_runs']}",
        "nNoiseSeeds": f"{summary['n_noise_seeds']}",
        "splitsPerNoise": f"{summary['splits_per_noise']}",
        "mCal": f"{summary['m_cal']}",
        "mCalSup": f"{summary['m_cal_sup']}",
        "mTrain": f"{summary['m_train']}",
        "calSlackPct": f"{summary['cal_slack_pct']:.2f}",
        "calSlackSupPct": f"{summary['cal_slack_sup_pct']:.2f}",
        "splitFit": f"{summary['m_fit']}",
        "splitCal": f"{summary['m_cal_sup']}",
        "splitTestMin": f"{summary['m_test_min']}",
        "splitTestMax": f"{summary['m_test_max']}",
        "baseFailPct": f"{heur['all_accept']['patient_risk_pct']:.1f}",
        "baseFailCiLo": f"{heur['all_accept']['patient_risk_ci_lo_pct']:.1f}",
        "baseFailCiHi": f"{heur['all_accept']['patient_risk_ci_hi_pct']:.1f}",
        "aucBsqi": f"{summary['feature_auroc']['1 - bSQI']:.3f}",
        "aucMorph": f"{summary['feature_auroc']['MorphDisp']:.3f}",
        "aucBas": f"{summary['feature_auroc']['1 - basSQI']:.3f}",
        "aucKurt": f"{summary['feature_auroc']['Kurt Deficit']:.3f}",
        "aucHf": f"{summary['feature_auroc']['HF Ratio']:.3f}",
        "aucCombined": f"{summary['score_auroc']:.3f}",
        "orphAccPct": f"{heur['orphanidou']['accept_pct']:.1f}",
        "orphRiskPct": f"{heur['orphanidou']['patient_risk_pct']:.2f}",
        "orphCleanEctPct": f"{csub['orphanidou']['clean_ectopic_accept']:.1f}",
        "cliffAccPct": f"{heur['clifford']['accept_pct']:.1f}",
        "cliffRiskPct": f"{heur['clifford']['patient_risk_pct']:.2f}",
        "cliffCleanEctPct": f"{csub['clifford']['clean_ectopic_accept']:.1f}",
        "crcZeroFiveAccPct": f"{cv['crc']['0.05']['accept_pct']:.1f}",
        "crcZeroFiveRiskPct": f"{cv['crc']['0.05']['patient_risk_pct']:.2f}",
        "crcZeroEightAccPct": f"{cv['crc']['0.08']['accept_pct']:.1f}",
        "crcZeroEightAbstPct": f"{cv['crc']['0.08']['abstain_pct']:.1f}",
        "crcZeroEightRiskPct": f"{cv['crc']['0.08']['patient_risk_pct']:.2f}",
        "crcZeroEightCiLo": f"{cv['crc']['0.08']['patient_risk_ci_lo_pct']:.2f}",
        "crcZeroEightCiHi": f"{cv['crc']['0.08']['patient_risk_ci_hi_pct']:.2f}",
        "crcZeroEightCondFail": f"{cv['crc']['0.08']['cond_fail_pct']:.2f}",
        "crcZeroEightPval": f"{cv['crc']['0.08']['p_viol']:.3f}",
        "crcZeroEightBeatFOne": f"{cv['crc']['0.08']['acc_beat_f1_med']:.3f}",
        "crcZeroEightEctRec": f"{cv['crc']['0.08']['acc_ect_recall_pct']:.1f}",
        "crcZeroEightEctWinAcc": f"{cv['crc']['0.08']['ect_win_accept_pct']:.1f}",
        "crcZeroEightRmssd": f"{cv['crc']['0.08']['acc_rmssd_err_med']:.1f}",
        "crcZeroEightPrd": f"{cv['crc']['0.08']['acc_qrs_prd_med']:.1f}",
        "crcTenAccPct": f"{cv['crc']['0.10']['accept_pct']:.1f}",
        "crcTenRiskPct": f"{cv['crc']['0.10']['patient_risk_pct']:.2f}",
        "crcTenCleanEctPct": f"{csub['crc_10']['clean_ectopic_accept']:.1f}",
        "crcMTwentySevenAccPct": f"{cv['crc_m27']['0.08']['accept_pct']:.1f}",
        "crcMTwentySevenRiskPct": f"{cv['crc_m27']['0.08']['patient_risk_pct']:.2f}",
        "crcSupZeroEightAccPct": f"{cv['crc_sup']['0.08']['accept_pct']:.1f}",
        "crcSupZeroEightRiskPct": f"{cv['crc_sup']['0.08']['patient_risk_pct']:.2f}",
        "crcSupZeroEightEctWinAcc": f"{cv['crc_sup']['0.08']['ect_win_accept_pct']:.1f}",
        "coreTwoTplAccPct": f"{cv['core_2tpl']['0.08']['accept_pct']:.1f}",
        "coreTwoTplRiskPct": f"{cv['core_2tpl']['0.08']['patient_risk_pct']:.2f}",
        "coreTwoTplEctPct": f"{cv['core_2tpl']['0.08']['ect_win_accept_pct']:.1f}",
        "scaleUnitAccPct": f"{u08['accept_pct']:.1f}",
        "scaleUnitRiskPct": f"{u08['patient_risk_pct']:.2f}",
        "iidZeroEightRiskPct": f"{cv['iid']['0.08']['patient_risk_pct']:.2f}",
        "iidZeroEightCiLo": f"{cv['iid']['0.08']['patient_risk_ci_lo_pct']:.2f}",
        "iidZeroEightCiHi": f"{cv['iid']['0.08']['patient_risk_ci_hi_pct']:.2f}",
        "iidZeroEightViolPct": f"{cv['iid_fold_viol_pct']['0.08']:.1f}",
        "iidZeroEightCondFail": f"{cv['iid']['0.08']['cond_fail_pct']:.2f}",
        "iidZeroEightPval": f"{cv['iid']['0.08']['p_viol']:.3f}",
        "iidSupZeroEightRiskPct": f"{cv['iid_sup']['0.08']['patient_risk_pct']:.2f}",
        "causalNextRiskPct": f"{cv['causal_next']['0.08']['patient_risk_pct']:.2f}",
        "causalNextAccPct": f"{cv['causal_next']['0.08']['accept_pct']:.1f}",
        "causalNextPval": f"{cv['causal_next']['0.08']['p_viol']:.3f}",
        "wavZeroEightRiskPct": f"{w08['patient_risk_pct']:.2f}",
        "wavZeroEightAccPct": f"{w08['accept_pct']:.1f}",
        "detHrvZeroEightRiskPct": f"{dh08['patient_risk_pct']:.2f}",
        "detHrvZeroEightAccPct": f"{dh08['accept_pct']:.1f}",
        "tauMedZeroEight": f"{summary['tau_08_median']:.2f}",
        "stressMaSixRiskPct": f"{stress[1]['crc08_risk_pct']:.2f}",
        "stressEmSixRiskPct": f"{stress[2]['crc08_risk_pct']:.2f}",
        "stressEmZeroRiskPct": f"{stress[3]['crc08_risk_pct']:.2f}",
        "stressEmZeroCiLo": f"{stress[3]['crc08_risk_ci_lo_pct']:.2f}",
        "stressEmZeroCiHi": f"{stress[3]['crc08_risk_ci_hi_pct']:.2f}",
        "stressEmZeroAbstPct": f"{stress[3]['crc08_abstain_pct']:.1f}",
        "stressEmZeroCliffRiskPct": f"{stress[3]['cliff_risk_pct']:.2f}",
        "stressEmZeroOrphAccPct": f"{stress[3]['orph_accept_pct']:.1f}",
        "stressEmZeroOrphRiskPct": f"{stress[3]['orph_risk_pct']:.2f}",
        "stressEmNegRiskPct": f"{stress[4]['crc08_risk_pct']:.2f}",
        "stressEmNegAbstPct": f"{stress[4]['crc08_abstain_pct']:.1f}",
        "lrZeroEightAccPct": f"{cv['ml_lr']['0.08']['accept_pct']:.1f}",
        "lrZeroEightRiskPct": f"{cv['ml_lr']['0.08']['patient_risk_pct']:.2f}",
        "lrZeroEightEctWinAcc": f"{cv['ml_lr']['0.08']['ect_win_accept_pct']:.1f}",
        "gbZeroEightAccPct": f"{cv['ml_gb']['0.08']['accept_pct']:.1f}",
        "gbZeroEightRiskPct": f"{cv['ml_gb']['0.08']['patient_risk_pct']:.2f}",
        "morphZeroEightAccPct": f"{cv['single_morph']['0.08']['accept_pct']:.1f}",
        "morphZeroEightEctWinAcc": f"{cv['single_morph']['0.08']['ect_win_accept_pct']:.1f}",
        "hfZeroEightCondFail": f"{cv['single_hf']['0.08']['cond_fail_pct']:.1f}",
        "orphCrcZeroEightAccPct": f"{cv['heur_orph']['0.08']['accept_pct']:.1f}",
        "orphCrcZeroEightRiskPct": f"{cv['heur_orph']['0.08']['patient_risk_pct']:.2f}",
        "orphCrcZeroEightEctPct": f"{cv['heur_orph']['0.08']['ect_win_accept_pct']:.1f}",
        "cliffCrcZeroEightAccPct": f"{cv['heur_cliff']['0.08']['accept_pct']:.1f}",
        "cliffCrcZeroEightRiskPct": f"{cv['heur_cliff']['0.08']['patient_risk_pct']:.2f}",
        "cliffCrcZeroEightEctPct": f"{cv['heur_cliff']['0.08']['ect_win_accept_pct']:.1f}",
        "mondTenSinusRisk": f"{mond['0.10']['Sinus']['mondrian_risk_pct']:.2f}",
        "mondTenSinusAcc": f"{mond['0.10']['Sinus']['mondrian_accept_pct']:.1f}",
        "mondTenEctRisk": f"{mond['0.10']['Ectopic']['mondrian_risk_pct']:.2f}",
        "mondTenEctAcc": f"{mond['0.10']['Ectopic']['mondrian_accept_pct']:.1f}",
        "margTenSinusRisk": f"{mond['0.10']['Sinus']['marginal_risk_pct']:.2f}",
        "margTenEctRisk": f"{mond['0.10']['Ectopic']['marginal_risk_pct']:.2f}",
        "margEightPacedRisk": f"{mond['0.08']['Paced']['marginal_risk_pct']:.2f}",
        "corrIidHomogViol": f"{corr[0]['iid_viol_pct']:.1f}",
        "corrRhoHighIidViol": f"{corr[-1]['iid_viol_pct']:.1f}",
        "corrRhoHighCrcViol": f"{corr[-1]['crc_viol_pct']:.1f}",
        "corrRhoHighIidPninetyfive": f"{corr[-1]['iid_risk_p95_pct']:.2f}",
        "corrRhoHighCrcPninetyfive": f"{corr[-1]['crc_risk_p95_pct']:.2f}",
        "runtimeMsPerWin": f"{summary.get('runtime_ms_per_window', 6.4):.1f}",
    }

    ltt = summary.get("conditional_ltt", {})
    if "0.08" in ltt:
        m["lttZeroEightAccPct"] = f"{ltt['0.08']['accept_pct']:.1f}"
        m["lttZeroEightEmpCondRisk"] = f"{ltt['0.08']['empirical_cond_risk_pct']:.2f}"
    if "0.10" in ltt:
        m["lttTenAccPct"] = f"{ltt['0.10']['accept_pct']:.1f}"
        m["lttTenEmpCondRisk"] = f"{ltt['0.10']['empirical_cond_risk_pct']:.2f}"

    dist = summary.get("seed0_distributions_08", {}).get("crc", {})
    if dist:
        m["crcZeroEightBeatFOneQTwentyFive"] = f"{dist['beat_f1']['p25']:.3f}"
        m["crcZeroEightBeatFOneQSeventyFive"] = f"{dist['beat_f1']['p75']:.3f}"
        m["crcZeroEightPrdQTwentyFive"] = f"{dist['qrs_prd']['p25']:.1f}"
        m["crcZeroEightPrdQSeventyFive"] = f"{dist['qrs_prd']['p75']:.1f}"
        m["crcZeroEightRmssdQTwentyFive"] = f"{dist['rmssd_err']['p25']:.1f}"
        m["crcZeroEightRmssdQSeventyFive"] = f"{dist['rmssd_err']['p75']:.1f}"

    macro_lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (LATEX_DIR / "results_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mitdb_dir = DATA_ROOT / "mitdb"
    nstdb_dir = DATA_ROOT / "nstdb"

    print("Loading MIT-BIH and NSTDB...")
    streams = load_mitdb_contiguous_streams(mitdb_dir, MITDB_RECORDS_48, n_strips_per_record=60, strip_s=10.0, start_s=60.0)
    nstdb_cal = load_nstdb_360hz(nstdb_dir, split="cal")
    nstdb_test = load_nstdb_360hz(nstdb_dir, split="test")

    n_noise_seeds = 5
    splits_per_noise = 4
    datasets_bp: list[dict[str, Any]] = []
    datasets_wav: list[dict[str, Any]] = []
    corpus_0: list[dict[str, Any]] = []
    t_proc_start = time.perf_counter()
    for n_idx in range(n_noise_seeds):
        corpus = generate_ambulatory_corpus(streams, nstdb_cal, seed=2026 + n_idx)
        if n_idx == 0:
            corpus_0 = corpus
        datasets_bp.append(_process_corpus(corpus, denoiser="bandpass"))
        datasets_wav.append(_process_corpus(corpus, denoiser="wavelet"))
        print(f"Processed noise seed {n_idx + 1}/{n_noise_seeds} ({time.time() - t0:.1f}s)")
    runtime_ms_per_win = ((time.perf_counter() - t_proc_start) * 1000.0) / float(2 * n_noise_seeds * len(corpus_0))

    data_bp_0 = datasets_bp[0]

    feat_auroc = {
        name: float(np.mean([roc_auc_score(d["loss_nom"], d["features"][:, j]) for d in datasets_bp]))
        for j, name in enumerate(FEAT_NAMES)
    }
    score_auroc = float(np.mean([roc_auc_score(d["loss_nom"], score_unsupervised(d["features"])) for d in datasets_bp]))

    print("Running 20-run 6-fold cross-validation...")
    heuristics = {
        "all_accept": _evaluate_fixed_rule(data_bp_0, np.ones(len(corpus_0), dtype=bool)),
        "orphanidou": _evaluate_fixed_rule(data_bp_0, data_bp_0["orph_accept"]),
        "clifford": _evaluate_fixed_rule(data_bp_0, data_bp_0["cliff_accept"]),
    }

    cv_raw = run_multi_seed_cv(datasets_bp, loss_key="loss_nom", splits_per_noise=splits_per_noise, n_folds=6, include_baselines=True)
    cv_wav_raw = run_multi_seed_cv(datasets_wav, loss_key="loss_nom", splits_per_noise=splits_per_noise, n_folds=6, include_baselines=False)
    cv_det_hrv_raw = run_multi_seed_cv(datasets_bp, loss_key="loss_det_hrv", splits_per_noise=splits_per_noise, n_folds=6, include_baselines=False)
    cv_strict_raw = run_multi_seed_cv(datasets_bp, loss_key="loss_strict", splits_per_noise=splits_per_noise, n_folds=6, include_baselines=False)
    cv_lenient_raw = run_multi_seed_cv(datasets_bp, loss_key="loss_lenient", splits_per_noise=splits_per_noise, n_folds=6, include_baselines=False)

    def _summarize_cv(raw_cv: dict[str, Any]) -> dict[str, Any]:
        agg: dict[str, Any] = {
            "crc": {},
            "crc_m27": {},
            "crc_sup": {},
            "iid": {},
            "iid_sup": {},
            "causal_next": {},
            "crc_fold_viol_pct": {},
            "iid_fold_viol_pct": {},
        }
        baseline_keys = (
            "single_bsqi",
            "single_morph",
            "single_bas",
            "single_hf",
            "single_kurt",
            "heur_orph",
            "core_2tpl",
            "heur_cliff",
            "scale_unit",
            "ml_lr",
            "ml_gb",
        )
        for bk in baseline_keys:
            if bk in raw_cv:
                agg[bk] = {}

        for a in ALPHAS:
            k = f"{a:.2f}"
            agg["crc"][k] = _aggregate_seed_list(raw_cv["crc"][a], alpha=a)
            agg["crc_m27"][k] = _aggregate_seed_list(raw_cv["crc_m27"][a], alpha=a)
            agg["crc_sup"][k] = _aggregate_seed_list(raw_cv["crc_sup"][a], alpha=a)
            agg["iid"][k] = _aggregate_seed_list(raw_cv["iid"][a], alpha=a)
            agg["iid_sup"][k] = _aggregate_seed_list(raw_cv["iid_sup"][a], alpha=a)
            agg["causal_next"][k] = _aggregate_seed_list(raw_cv["causal_next"][a], alpha=a)
            for bk in baseline_keys:
                if bk in raw_cv:
                    agg[bk][k] = _aggregate_seed_list(raw_cv[bk][a], alpha=a)
            agg["crc_fold_viol_pct"][k] = float(raw_cv["crc_fold_violations"][a]) / float(raw_cv["total_folds"]) * 100.0
            agg["iid_fold_viol_pct"][k] = float(raw_cv["iid_fold_violations"][a]) / float(raw_cv["total_folds"]) * 100.0
        return agg

    cv_agg = _summarize_cv(cv_raw)
    wave_agg = _summarize_cv(cv_wav_raw)
    det_hrv_agg = _summarize_cv(cv_det_hrv_raw)
    strict_agg = _summarize_cv(cv_strict_raw)
    lenient_agg = _summarize_cv(cv_lenient_raw)

    print("Running LTT, Mondrian subgroup CRC, and correlation sweep...")
    ltt_results = run_conditional_ltt_study(data_bp_0, n_seeds=10)
    subgroup_mondrian = run_subgroup_mondrian_study(datasets_bp, n_folds=6)
    correlation_sweep = simulate_clustered_correlation_sweep(m_cal=cv_raw["m_cal"], m_test=50, n_trials=300, alpha=0.08, seed=2026)

    seed0_acc_10 = cv_raw["seed0_accept_10"]
    seed0_acc_08 = cv_raw["seed0_accept_08"]
    is_sinus_clean = data_bp_0["is_clean"] & (data_bp_0["group"] == "Sinus")
    is_ect_clean = data_bp_0["is_clean"] & data_bp_0["has_ectopic"]
    is_sinus = data_bp_0["group"] == "Sinus"
    is_ect_grp = data_bp_0["group"] == "Ectopic"
    is_paced = data_bp_0["group"] == "Paced"

    cohort_subgroups = {}
    for name, acc_m in (("crc_08", seed0_acc_08), ("crc_10", seed0_acc_10), ("orphanidou", data_bp_0["orph_accept"]), ("clifford", data_bp_0["cliff_accept"])):
        cohort_subgroups[name] = {
            "clean_sinus_accept": float(np.mean(acc_m[is_sinus_clean])) * 100.0,
            "clean_ectopic_accept": float(np.mean(acc_m[is_ect_clean])) * 100.0,
            "risk_sinus": float(np.mean(_patient_risks_for_mask(acc_m, data_bp_0["loss_nom"], data_bp_0["pid"], is_sinus))) * 100.0,
            "risk_ectopic": float(np.mean(_patient_risks_for_mask(acc_m, data_bp_0["loss_nom"], data_bp_0["pid"], is_ect_grp))) * 100.0,
            "risk_paced": float(np.mean(_patient_risks_for_mask(acc_m, data_bp_0["loss_nom"], data_bp_0["pid"], is_paced))) * 100.0,
        }

    print("Running NSTDB stress tests...")
    tau_08_med = float(np.median(cv_raw["tau_list"][0.08]))
    tau_medians = {f"{a:.2f}": float(np.median(cv_raw["tau_list"][a])) for a in ALPHAS}
    stress_specs = [
        ("Clean Base", "Clean Base", None, np.inf),
        ("ma 6dB", r"NSTDB \texttt{ma} (6\,dB)", "ma", 6.0),
        ("em 6dB", r"NSTDB \texttt{em} (6\,dB)", "em", 6.0),
        ("em 0dB", r"NSTDB \texttt{em} (0\,dB)", "em", 0.0),
        ("em -3dB", r"NSTDB \texttt{em} (-3\,dB)", "em", -3.0),
    ]
    stress_rows = []
    for short_lbl, tex_lbl, nkey, snr in stress_specs:
        if nkey is None:
            sc_corpus = [dict(r, sig_obs_mv=r["sig_clean_mv"], noise_regime="clean", snr_db=np.inf) for r in streams]
        else:
            sc_corpus = inject_stress_noise(streams, nstdb_test, noise_key=nkey, snr_db=snr, seed=999)
        s_data = _process_corpus(sc_corpus, denoiser="bandpass")
        s_scores = score_unsupervised(s_data["features"])
        s_acc = np.zeros(len(sc_corpus), dtype=bool)
        for fdict in cv_raw["seed0_folds_08"]:
            pmask = np.isin(s_data["pid"], fdict["test_pids"])
            s_acc[pmask] = s_scores[pmask] <= fdict["tau"]
        s_m = evaluate_gating_decision(
            s_acc,
            s_data["loss_nom"],
            s_data["beat_f1"],
            s_data["rmssd_err_ms"],
            s_data["qrs_prd_pct"],
            s_data["ectopic_recall"],
        )
        pat_r = _patient_risks_for_mask(s_acc, s_data["loss_nom"], s_data["pid"])
        rm, rlo, rhi = patient_cluster_bootstrap_ci(pat_r, n_boot=2000, seed=42)

        c_acc = s_data["cliff_accept"]
        c_pat_r = _patient_risks_for_mask(c_acc, s_data["loss_nom"], s_data["pid"])
        c_rm, c_rlo, c_rhi = patient_cluster_bootstrap_ci(c_pat_r, n_boot=2000, seed=42)

        o_acc = s_data["orph_accept"]
        o_pat_r = _patient_risks_for_mask(o_acc, s_data["loss_nom"], s_data["pid"])
        o_rm, o_rlo, o_rhi = patient_cluster_bootstrap_ci(o_pat_r, n_boot=2000, seed=42)

        stress_rows.append(
            {
                "label": short_lbl,
                "tex_label": tex_lbl,
                "n_windows": len(sc_corpus),
                "crc08_accept_pct": s_m["accept_pct"],
                "crc08_abstain_pct": s_m["abstain_pct"],
                "crc08_risk_pct": rm * 100.0,
                "crc08_risk_ci_lo_pct": rlo * 100.0,
                "crc08_risk_ci_hi_pct": rhi * 100.0,
                "crc08_beat_f1_med": s_m["acc_beat_f1_med"] if np.isfinite(s_m["acc_beat_f1_med"]) else 0.0,
                "crc08_qrs_prd_med": s_m["acc_qrs_prd_med"] if np.isfinite(s_m["acc_qrs_prd_med"]) else 0.0,
                "cliff_accept_pct": float(np.mean(c_acc)) * 100.0,
                "cliff_abstain_pct": 100.0 - float(np.mean(c_acc)) * 100.0,
                "cliff_risk_pct": c_rm * 100.0,
                "cliff_risk_ci_lo_pct": c_rlo * 100.0,
                "cliff_risk_ci_hi_pct": c_rhi * 100.0,
                "orph_accept_pct": float(np.mean(o_acc)) * 100.0,
                "orph_abstain_pct": 100.0 - float(np.mean(o_acc)) * 100.0,
                "orph_risk_pct": o_rm * 100.0,
                "orph_risk_ci_lo_pct": o_rlo * 100.0,
                "orph_risk_ci_hi_pct": o_rhi * 100.0,
            }
        )

    frozen_params = {
        "description": "Auto-generated unsupervised fixed-scale composite score parameters and MIT-BIH median CRC thresholds.",
        "medians": [0.0, 0.0, 0.0, 0.0, 0.0],
        "iqrs": list(_CRC_CFG.fixed_unsup_scales),
        "weights": list(_CRC_CFG.fixed_unsup_weights),
        "tau_medians": tau_medians,
        "supervised_reference": {
            "medians": cv_raw["sup_medians_mean"],
            "iqrs": cv_raw["sup_scales_mean"],
            "weights": cv_raw["sup_weights_mean"],
            "tau_medians": {f"{a:.2f}": float(np.median(cv_raw["tau_sup_list"][a])) for a in ALPHAS},
        },
    }
    (RESULTS_DIR / "frozen_score_params.json").write_text(json.dumps(frozen_params, indent=2), encoding="utf-8")

    summary = {
        "n_patients": int(len(np.unique(data_bp_0["pid"]))),
        "n_windows": int(len(corpus_0)),
        "n_runs": cv_raw["n_runs"],
        "n_noise_seeds": cv_raw["n_noise_seeds"],
        "splits_per_noise": cv_raw["splits_per_noise"],
        "m_fit": cv_raw["m_fit"],
        "m_cal": cv_raw["m_cal"],
        "m_cal_sup": cv_raw["m_cal_sup"],
        "m_train": cv_raw["m_train"],
        "m_test_min": cv_raw["m_test_min"],
        "m_test_max": cv_raw["m_test_max"],
        "cal_slack_pct": cv_raw["cal_slack_pct"],
        "cal_slack_sup_pct": cv_raw["cal_slack_sup_pct"],
        "runtime_ms_per_window": float(runtime_ms_per_win),
        "tau_08_median": tau_08_med,
        "tau_medians": tau_medians,
        "weights_mean": cv_raw["sup_weights_mean"],
        "feature_auroc": feat_auroc,
        "score_auroc": score_auroc,
        "heuristics": heuristics,
        "cv_nominal": cv_agg,
        "cv_wavelet": wave_agg,
        "sensitivity": {"det_hrv": det_hrv_agg, "strict": strict_agg, "lenient": lenient_agg},
        "cohort_subgroups": cohort_subgroups,
        "subgroup_mondrian": subgroup_mondrian,
        "correlation_sweep": correlation_sweep,
        "stress_nstdb": stress_rows,
        "conditional_ltt": ltt_results,
        "seed0_distributions_08": cv_raw.get("seed0_distributions_08", {}),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_tables_and_macros(summary)
    import importlib
    try:
        import draw_all_figures as daf
    except ImportError:
        import experiments.draw_all_figures as daf
    importlib.reload(daf)
    daf.draw_figure_1()
    daf.draw_figure_2()
    daf.draw_figure_3()
    daf.draw_figure_4()
    print(f"Benchmark complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
