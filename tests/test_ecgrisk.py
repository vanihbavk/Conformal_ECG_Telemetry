"""Unit tests for ambulatory ECG quality gating and patient-cluster CRC."""

from __future__ import annotations

import unittest

import numpy as np

from ecgrisk.afe import apply_diagnostic_afe_360hz, block_haar_swt_denoise, causal_clinical_bandpass_360hz
from ecgrisk.config import AfeConfig, ConformalConfig, MetricsConfig, SqiConfig
from ecgrisk.conformal import (
    calibrate_crc_threshold,
    calibrate_group_crc_thresholds,
    calibrate_iid_threshold,
    evaluate_gating_decision,
    fit_robust_score_params,
    patient_cluster_bootstrap_ci,
    patient_clustered_validity_test,
    score_uncertainty,
    score_unsupervised,
    simulate_clustered_correlation_sweep,
    wilson_binomial_ci,
)
from ecgrisk.datasets import (
    _scale_noise_to_snr,
    classify_patient_group,
    generate_ambulatory_corpus,
    inject_stress_noise,
)
from ecgrisk.metrics import compute_qrs_morphology_prd, compute_rmssd_ms, evaluate_clinical_window, match_beats_aami
from ecgrisk.sqi import compute_two_template_morph_disp, detect_qrs_pantompkins, extract_sqi_features


def _synth_qrs_template(fs: float = 360.0, width_s: float = 0.020, sign: float = 1.0) -> np.ndarray:
    t = np.arange(-int(0.1 * fs), int(0.1 * fs) + 1) / fs
    return sign * 1.4 * np.exp(-0.5 * (t / width_s) ** 2) - sign * 0.25 * np.exp(-0.5 * ((t - 0.03) / 0.015) ** 2)


def _synth_strip(fs: float = 360.0, dur_s: float = 10.0, bigeminy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(fs * dur_s))
    sig = np.zeros(n, dtype=float)
    qrs_norm = _synth_qrs_template(fs=fs, width_s=0.014, sign=1.0)
    qrs_pvc = _synth_qrs_template(fs=fs, width_s=0.026, sign=-0.9)
    half = len(qrs_norm) // 2

    peaks = []
    t_s = 0.65
    idx = 0
    while t_s < dur_s - 0.35:
        peaks.append(t_s)
        c = int(round(t_s * fs))
        tpl = qrs_pvc if (bigeminy and (idx % 2 == 1)) else qrs_norm
        lo, hi = c - half, c + half + 1
        if lo >= 0 and hi <= n:
            sig[lo:hi] += tpl
        t_s += 0.48 if (bigeminy and idx % 2 == 0) else (1.15 if bigeminy else 0.85)
        idx += 1
    return sig, np.asarray(peaks, dtype=float)


class TestEcgRisk(unittest.TestCase):
    def test_afe_wavelet_and_sinus_detection(self) -> None:
        raw, peaks_true = _synth_strip(bigeminy=False)
        afe = apply_diagnostic_afe_360hz(raw, fs=360.0)
        clin = causal_clinical_bandpass_360hz(afe, fs=360.0)
        wav = block_haar_swt_denoise(afe, fs=360.0)
        self.assertEqual(len(clin), len(raw))
        self.assertEqual(len(wav), len(raw))

        est_peaks = detect_qrs_pantompkins(clin, fs=360.0)
        res = evaluate_clinical_window(clin, clin, peaks_true, est_peaks, fs=360.0)
        self.assertGreaterEqual(res["beat_f1"], 0.95)
        self.assertEqual(res["clin_fail"], 0.0)
        self.assertEqual(res["clin_fail_det_hrv"], 0.0)

    def test_bigeminy_preserves_morph_disp_while_orphanidou_rejects(self) -> None:
        raw, peaks_true = _synth_strip(bigeminy=True)
        afe = apply_diagnostic_afe_360hz(raw, fs=360.0)
        clin = causal_clinical_bandpass_360hz(afe, fs=360.0)
        sqi = extract_sqi_features(afe, clin, fs=360.0)
        self.assertFalse(sqi["orphanidou_accept"])
        morph_disp, _ = compute_two_template_morph_disp(clin, peaks_true, fs=360.0)
        self.assertLess(morph_disp, 0.20)

    def test_noise_degrades_sqi_features(self) -> None:
        raw, _ = _synth_strip(bigeminy=False)
        rng = np.random.default_rng(7)
        noisy = raw + rng.normal(0.0, 1.5, size=len(raw))

        afe_c = apply_diagnostic_afe_360hz(raw)
        sqi_clean = extract_sqi_features(afe_c, causal_clinical_bandpass_360hz(afe_c))
        afe_n = apply_diagnostic_afe_360hz(noisy)
        sqi_noisy = extract_sqi_features(afe_n, causal_clinical_bandpass_360hz(afe_n))
        self.assertGreater(float(np.sum(sqi_noisy["features"])), float(np.sum(sqi_clean["features"])) + 0.5)

    def test_crc_patient_exchangeability_and_iqr_scale(self) -> None:
        feats = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=float)
        med, scale, w = fit_robust_score_params(feats, losses_target=None)
        np.testing.assert_allclose(med, [2.0, 4.0])
        np.testing.assert_allclose(scale, [2.0, 4.0])
        np.testing.assert_allclose(w, [0.5, 0.5])

        rng = np.random.default_rng(42)
        n_patients = 28
        cal_scores, cal_losses, cal_pids = [], [], []
        for p in range(n_patients):
            duty = 0.85 if p < 4 else 0.05
            for _ in range(30):
                fail = float(rng.random() < duty)
                sc = float(rng.normal(2.5 if fail else 0.5, 0.4))
                cal_scores.append(sc)
                cal_losses.append(fail)
                cal_pids.append(f"p{p}")
        cs = np.asarray(cal_scores)
        cl = np.asarray(cal_losses)
        cp = np.asarray(cal_pids)
        tau_crc = calibrate_crc_threshold(cs, cl, cp, alpha=0.10)
        tau_iid = calibrate_iid_threshold(cs, cl, alpha=0.10)
        self.assertGreater(tau_crc, -np.inf)
        pat_risks = [float(np.mean(np.where(cs[cp == f"p{p}"] <= tau_crc, cl[cp == f"p{p}"], 0.0))) for p in range(n_patients)]
        self.assertLessEqual((sum(pat_risks) + 1.0) / (n_patients + 1.0), 0.10 + 1e-9)
        self.assertGreaterEqual(tau_iid, -np.inf)

        mean_val, lo, hi = patient_cluster_bootstrap_ci(pat_risks, n_boot=200)
        self.assertLessEqual(lo, mean_val)
        self.assertGreaterEqual(hi, mean_val)
        _ = score_uncertainty(feats, med, scale, w)

        # Unsupervised fixed-scale scoring sanity
        u_feats = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.50, 0.50]], dtype=float)
        u_sc = score_unsupervised(u_feats)
        np.testing.assert_allclose(u_sc, [0.0, 1.0])

        # Negative-valued score calibration sentinel sanity
        neg_cs = cs - 10.0
        tau_neg = calibrate_crc_threshold(neg_cs, cl, cp, alpha=0.10)
        self.assertAlmostEqual(tau_neg, tau_crc - 10.0, places=6)

    def test_config_wiring_actually_controls_behavior(self) -> None:
        raw, peaks_true = _synth_strip(bigeminy=False)
        afe_12 = apply_diagnostic_afe_360hz(raw, cfg=AfeConfig(adc_bits=12))
        afe_4 = apply_diagnostic_afe_360hz(raw, cfg=AfeConfig(adc_bits=4))
        self.assertGreater(float(np.max(np.abs(afe_12 - afe_4))), 0.01)

        clin = causal_clinical_bandpass_360hz(afe_12)
        sqi_a = extract_sqi_features(afe_12, clin, cfg=SqiConfig(kurt_target=6.0))
        sqi_b = extract_sqi_features(afe_12, clin, cfg=SqiConfig(kurt_target=20.0))
        self.assertNotAlmostEqual(float(sqi_a["features"][4]), float(sqi_b["features"][4]))

        feats = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0, 10.0]])
        sc_clip2 = score_unsupervised(feats, cfg=ConformalConfig(z_clip_max=2.0))
        sc_clip6 = score_unsupervised(feats, cfg=ConformalConfig(z_clip_max=6.0))
        self.assertAlmostEqual(float(sc_clip2[1]), 2.0)
        self.assertAlmostEqual(float(sc_clip6[1]), 6.0)

        # MetricsConfig controls PRD failure vs PRD-free det_hrv
        noisy_clin = clin + 0.8 * np.sin(np.linspace(0, 40, len(clin)))
        m_strict = evaluate_clinical_window(noisy_clin, clin, peaks_true, peaks_true, cfg=MetricsConfig(max_qrs_prd=1.0))
        self.assertEqual(m_strict["clin_fail"], 1.0)
        self.assertEqual(m_strict["clin_fail_det_hrv"], 0.0)

    def test_mondrian_group_crc_and_infeasibility(self) -> None:
        rng = np.random.default_rng(77)
        scores, losses, pids, groups = [], [], [], []
        # Group A: 20 patients (feasible at alpha=0.08 since 1/21 = 4.76%)
        for p in range(20):
            for _ in range(20):
                fail = float(rng.random() < 0.25)
                scores.append(float(rng.normal(2.0 if fail else 0.4, 0.3)))
                losses.append(fail)
                pids.append(f"A_{p}")
                groups.append("Ectopic")
        # Group B: 4 patients (infeasible at alpha=0.08 since 1/5 = 20%)
        for p in range(4):
            for _ in range(20):
                scores.append(0.3)
                losses.append(0.0)
                pids.append(f"B_{p}")
                groups.append("Paced")

        res = calibrate_group_crc_thresholds(
            np.asarray(scores), np.asarray(losses), np.asarray(pids), np.asarray(groups), alpha=0.08
        )
        self.assertTrue(res["Ectopic"]["feasible"])
        self.assertGreater(res["Ectopic"]["tau"], -np.inf)
        self.assertFalse(res["Paced"]["feasible"])
        self.assertEqual(res["Paced"]["tau"], -np.inf)

    def test_datasets_corpus_and_noise_injection(self) -> None:
        self.assertEqual(classify_patient_group("102", ["/", "N"]), "Paced")
        self.assertEqual(classify_patient_group("200", ["V"] * 5 + ["N"] * 20), "Ectopic")
        self.assertEqual(classify_patient_group("100", ["N"] * 50), "Sinus")

        raw, peaks = _synth_strip(bigeminy=False)
        unit_noise = np.random.default_rng(1).normal(0.0, 1.0, size=36000)
        scaled = _scale_noise_to_snr(raw, unit_noise[: len(raw)], snr_db=6.0)
        self.assertEqual(len(scaled), len(raw))

        streams = [
            {
                "pid": f"p{p}",
                "rec_id": f"p{p}",
                "group": "Sinus",
                "strip_idx": w,
                "fs": 360.0,
                "sig_clean_mv": raw,
                "r_peaks_s": peaks,
                "r_syms": ["N"] * len(peaks),
                "ect_flags": np.zeros(len(peaks), dtype=bool),
                "has_ectopic": False,
            }
            for p in range(4)
            for w in range(10)
        ]
        nstdb = {"bw": unit_noise, "em": unit_noise, "ma": unit_noise}
        corpus = generate_ambulatory_corpus(streams, nstdb, seed=10)
        self.assertEqual(len(corpus), len(streams))
        snrs = [r["snr_db"] for r in corpus if r["noise_regime"] != "clean"]
        self.assertTrue(all(-2.0 <= s <= 12.0 for s in snrs))

        stressed = inject_stress_noise(streams[:5], nstdb, noise_key="em", snr_db=0.0, seed=5)
        self.assertEqual(len(stressed), 5)
        self.assertTrue(all(r["noise_regime"] == "em_0dB" for r in stressed))

    def test_evaluation_stats_and_correlation_sweep(self) -> None:
        accept = np.array([True, True, False, True])
        losses = np.array([0.0, 1.0, 1.0, 0.0])
        f1 = np.array([1.0, 0.8, 0.5, 1.0])
        rmssd = np.array([2.0, 25.0, 50.0, 1.0])
        prd = np.array([0.0, 30.0, 40.0, 0.0])
        ect = np.array([1.0, np.nan, 0.0, 1.0])
        g = evaluate_gating_decision(accept, losses, f1, rmssd, prd, ect)
        self.assertAlmostEqual(g["accept_pct"], 75.0)
        self.assertAlmostEqual(g["silent_risk"], 0.25)

        lo, hi = wilson_binomial_ci(20, 100)
        self.assertLess(lo, 0.20)
        self.assertGreater(hi, 0.20)

        vt = patient_clustered_validity_test([0.04, 0.05, 0.06, 0.05, 0.04], alpha=0.08)
        self.assertGreater(vt["p_viol"], 0.90)

        sweep = simulate_clustered_correlation_sweep(persistence_grid=(0.0, 0.90), m_cal=25, m_test=25, n_trials=40, seed=11)
        self.assertEqual(len(sweep), 3)
        self.assertEqual(sweep[0]["regime"], "iid_homog")

    def test_bisection_calibration_matches_exact_grid(self) -> None:
        rng = np.random.default_rng(123)
        n_patients = 15
        cal_scores, cal_losses, cal_pids = [], [], []
        for p in range(n_patients):
            for _ in range(20):
                fail = float(rng.random() < 0.25)
                cal_scores.append(float(rng.uniform(0.0, 5.0)))
                cal_losses.append(fail)
                cal_pids.append(f"pat_{p}")
        cs = np.asarray(cal_scores)
        cl = np.asarray(cal_losses)
        cp = np.asarray(cal_pids)

        tau_bisect = calibrate_crc_threshold(cs, cl, cp, alpha=0.08)
        grid = np.linspace(float(cs.min()) - 0.1, float(cs.max()) + 0.1, 2000)
        valid_taus = []
        for t in grid:
            p_risks = [float(np.mean(np.where(cs[cp == f"pat_{p}"] <= t, cl[cp == f"pat_{p}"], 0.0))) for p in range(n_patients)]
            if (sum(p_risks) + 1.0) / (n_patients + 1.0) <= 0.08:
                valid_taus.append(t)
        if valid_taus:
            tau_grid = max(valid_taus)
            self.assertAlmostEqual(tau_bisect, tau_grid, delta=0.05)

    def test_calibrate_crc_threshold_tied_and_constant_scores(self) -> None:
        rng = np.random.default_rng(999)
        n_patients = 20
        cs_const = np.ones(100, dtype=float)
        cl_const = (rng.random(100) < 0.30).astype(float)
        cp_const = np.array([f"p_{i % n_patients}" for i in range(100)])

        for alpha in (0.05, 0.08, 0.10, 0.25, 0.50):
            tau = calibrate_crc_threshold(cs_const, cl_const, cp_const, alpha=alpha)
            unique_p = np.unique(cp_const)
            p_risks = [float(np.mean(np.where(cs_const[cp_const == p] <= tau, cl_const[cp_const == p], 0.0))) for p in unique_p]
            empirical_bound = (sum(p_risks) + 1.0) / (len(unique_p) + 1.0)
            self.assertLessEqual(empirical_bound, alpha + 1e-12)

        cs_tied = rng.choice([0.1, 0.5, 0.9], size=300)
        cl_tied = (rng.random(300) < 0.25).astype(float)
        cp_tied = np.array([f"p_{i % 30}" for i in range(300)])

        for alpha in (0.05, 0.08, 0.10, 0.15, 0.25):
            tau = calibrate_crc_threshold(cs_tied, cl_tied, cp_tied, alpha=alpha)
            unique_p = np.unique(cp_tied)
            p_risks = [float(np.mean(np.where(cs_tied[cp_tied == p] <= tau, cl_tied[cp_tied == p], 0.0))) for p in unique_p]
            empirical_bound = (sum(p_risks) + 1.0) / (len(unique_p) + 1.0)
            self.assertLessEqual(empirical_bound, alpha + 1e-12)

    def test_metrics_without_wfdb(self) -> None:
        ref_peaks = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        det_peaks = np.array([1.01, 1.99, 3.02, 4.00, 5.01], dtype=float)
        m = match_beats_aami(ref_peaks, det_peaks, tol_s=0.075)
        self.assertEqual(m["tp"], 5.0)
        self.assertEqual(m["beat_f1"], 1.0)

        rmssd = compute_rmssd_ms(det_peaks)
        self.assertGreaterEqual(rmssd, 0.0)

        sig_ref = np.sin(np.linspace(0, 10, 100))
        sig_noisy = sig_ref + 0.05 * np.cos(np.linspace(0, 10, 100))
        prd_res = compute_qrs_morphology_prd(sig_noisy, sig_ref, np.array([0.5]), fs=100.0)
        self.assertGreaterEqual(prd_res["qrs_prd_pct"], 0.0)
        self.assertLess(prd_res["qrs_prd_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
