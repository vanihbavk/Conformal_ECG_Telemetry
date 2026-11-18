"""Cross-regime unseen-noise stress test: calibrate on bw + ma, test on unseen em (6, 0, -3 dB) + target recalibration recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ecgrisk.afe import apply_diagnostic_afe_360hz, causal_clinical_bandpass_360hz
from ecgrisk.conformal import calibrate_crc_threshold, patient_cluster_bootstrap_ci, score_uncertainty
from ecgrisk.datasets import (
    MITDB_RECORDS_48,
    generate_ambulatory_corpus,
    inject_stress_noise,
    load_mitdb_contiguous_streams,
    load_nstdb_360hz,
)
from ecgrisk.metrics import evaluate_clinical_window
from ecgrisk.sqi import extract_sqi_features


def _extract_features_and_loss(corpus: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(corpus)
    feats = np.zeros((n, 5), dtype=float)
    loss = np.zeros(n, dtype=float)
    pids = []
    for idx, row in enumerate(corpus):
        fs = float(row["fs"])
        afe_clean = apply_diagnostic_afe_360hz(row["sig_clean_mv"], fs=fs)
        afe_obs = apply_diagnostic_afe_360hz(row["sig_obs_mv"], fs=fs)
        clin_clean = causal_clinical_bandpass_360hz(afe_clean, fs=fs)
        clin_obs = causal_clinical_bandpass_360hz(afe_obs, fs=fs)
        sqi = extract_sqi_features(afe_obs, clin_obs, fs=fs)
        cm = evaluate_clinical_window(
            clin_obs, clin_clean, row["r_peaks_s"], sqi["peaks_a_s"], ect_flags=row["ect_flags"], fs=fs
        )
        feats[idx] = sqi["features"]
        loss[idx] = cm["clin_fail"]
        pids.append(row["pid"])
    return feats, loss, np.asarray(pids)


def run_unseen_noise_experiment() -> None:
    mitdb_dir = DATA_ROOT / "mitdb"
    nstdb_dir = DATA_ROOT / "nstdb"

    print("Loading MIT-BIH & NSTDB...")
    streams = load_mitdb_contiguous_streams(mitdb_dir, MITDB_RECORDS_48, n_strips_per_record=60, strip_s=10.0, start_s=60.0)
    nstdb_cal = load_nstdb_360hz(nstdb_dir, split="cal")
    nstdb_test = load_nstdb_360hz(nstdb_dir, split="test")

    print("Generating calibration corpus with purely bw and ma (NO em)...")
    cal_corpus = generate_ambulatory_corpus(streams, nstdb_cal, seed=2026, noise_regimes=("bw", "ma"))
    cal_feats, cal_loss, cal_pids = _extract_features_and_loss(cal_corpus)

    frozen_path = ROOT / "results" / "frozen_score_params.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    med = np.asarray(frozen["medians"], dtype=float)
    scale = np.asarray(frozen["iqrs"], dtype=float)
    w = np.asarray(frozen["weights"], dtype=float)

    cal_scores = score_uncertainty(cal_feats, med, scale, w)
    unique_pids = sorted(set(cal_pids.tolist()))
    m_cal = len(unique_pids)
    tau_08 = calibrate_crc_threshold(cal_scores, cal_loss, cal_pids, alpha=0.08)
    print(f"Calibrated tau on bw+ma (M={m_cal} patients) at alpha=0.08: tau={tau_08:.4f}")

    results: dict[str, dict] = {}
    em0_cache = None
    for snr in [6.0, 0.0, -3.0]:
        print(f"\nEvaluating unseen noise: em at {snr} dB...")
        test_corpus = inject_stress_noise(streams, nstdb_test, noise_key="em", snr_db=snr, seed=999)
        test_feats, test_loss, test_pids = _extract_features_and_loss(test_corpus)
        test_scores = score_uncertainty(test_feats, med, scale, w)
        if abs(snr - 0.0) < 1e-9:
            em0_cache = (test_scores, test_loss, test_pids)

        accept = test_scores <= tau_08
        acc_pct = float(np.mean(accept) * 100.0)
        p_risks = [float(np.mean(accept[test_pids == p] * test_loss[test_pids == p])) for p in unique_pids]
        rm, rlo, rhi = patient_cluster_bootstrap_ci(p_risks, n_boot=2000, seed=42)
        results[f"em_{int(snr)}dB"] = {
            "snr_db": snr,
            "accept_pct": acc_pct,
            "risk_pct": rm * 100.0,
            "risk_ci_lo": rlo * 100.0,
            "risk_ci_hi": rhi * 100.0,
            "exceeds_alpha_08": bool(rm > 0.08),
        }
        print(f"  Accept: {acc_pct:.1f}% | Patient Risk: {rm*100.0:.2f}% [{rlo*100.0:.2f}%, {rhi*100.0:.2f}%]")

    # Target recalibration recovery curve on em 0 dB across target calibration sizes M_target
    recovery: dict[str, dict] = {}
    if em0_cache is not None:
        s_em0, l_em0, p_em0 = em0_cache
        pid_arr = np.asarray(unique_pids)
        rng = np.random.default_rng(2026)
        for m_targ in (15, 25, 35):
            r_vals = []
            a_vals = []
            for _ in range(30):
                perm = rng.permutation(pid_arr)
                cal_set = set(perm[:m_targ].tolist())
                cal_m = np.isin(p_em0, list(cal_set))
                test_m = ~cal_m
                tau_t = calibrate_crc_threshold(s_em0[cal_m], l_em0[cal_m], p_em0[cal_m], alpha=0.08)
                acc_t = s_em0[test_m] <= tau_t
                a_vals.append(float(np.mean(acc_t)) * 100.0)
                test_pids_sub = [p for p in unique_pids if p not in cal_set]
                pr = [float(np.mean(np.where(s_em0[p_em0 == p] <= tau_t, l_em0[p_em0 == p], 0.0))) for p in test_pids_sub]
                r_vals.append(float(np.mean(pr)) * 100.0)
            recovery[str(m_targ)] = {
                "m_target": m_targ,
                "accept_pct": float(np.mean(a_vals)),
                "risk_pct": float(np.mean(r_vals)),
            }

    out_file = ROOT / "results" / "unseen_noise_results.json"
    out_file.write_text(json.dumps({"tau_08": tau_08, "results": results, "em_0dB_recalibration": recovery}, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    run_unseen_noise_experiment()
