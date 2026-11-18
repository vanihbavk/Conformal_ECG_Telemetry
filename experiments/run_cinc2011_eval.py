#!/usr/bin/env python3
"""External evaluation on PhysioNet/CinC Challenge 2011 Set A mobile-phone ECGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import wfdb
except ImportError:
    wfdb = None

from ecgrisk.afe import apply_diagnostic_afe_360hz, causal_clinical_bandpass_360hz
from ecgrisk.conformal import calibrate_crc_threshold, score_uncertainty, wilson_binomial_ci
from ecgrisk.sqi import extract_sqi_features

ALPHAS = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    """Return (proportion, ci_lo, ci_hi) via shared wilson_binomial_ci."""
    if n == 0:
        return 0.0, 0.0, 0.0
    lo, hi = wilson_binomial_ci(k, n)
    return float(k / n), lo, hi


def load_cinc2011_records(
    cinc_dir: Path,
    frozen_params: dict[str, Any],
) -> list[dict[str, Any]]:
    if wfdb is None:
        raise ImportError("wfdb is required to load CinC 2011 records. Install via pip install wfdb.")

    seta_dir = cinc_dir / "set-a"
    acc_file = cinc_dir / "RECORDS-acceptable"
    unacc_file = cinc_dir / "RECORDS-unacceptable"
    if not acc_file.exists() or not unacc_file.exists():
        raise FileNotFoundError(
            f"Missing RECORDS-acceptable or RECORDS-unacceptable in {cinc_dir}. "
            "Run `python3 experiments/download_data.py --datasets cinc2011` first."
        )

    acc_set = set(line.strip() for line in acc_file.read_text().splitlines() if line.strip())
    unacc_set = set(line.strip() for line in unacc_file.read_text().splitlines() if line.strip())

    med = np.array(frozen_params["medians"], dtype=float)
    iqr = np.array(frozen_params["iqrs"], dtype=float)
    w = np.array(frozen_params["weights"], dtype=float)

    all_recs = sorted(acc_set | unacc_set)
    records = []
    t0 = time.time()
    for rec_id in all_recs:
        record_path = seta_dir / rec_id
        if not (record_path.with_suffix(".hea").exists() and record_path.with_suffix(".dat").exists()):
            continue
        rec = wfdb.rdrecord(str(record_path))
        lead_names = [s.upper() for s in rec.sig_name]
        if "II" in lead_names:
            lead_idx = lead_names.index("II")
            sig_500 = rec.p_signal[:, lead_idx].astype(float)
        elif "MLII" in lead_names:
            lead_idx = lead_names.index("MLII")
            sig_500 = rec.p_signal[:, lead_idx].astype(float)
        else:
            sig_500 = rec.p_signal[:, 0].astype(float)

        if np.any(np.isnan(sig_500)):
            sig_500 = np.nan_to_num(sig_500, nan=0.0)

        # Resample 500 Hz -> 360 Hz (360/500 = 18/25)
        sig_360 = resample_poly(sig_500, 18, 25)

        afe = apply_diagnostic_afe_360hz(sig_360, fs=360.0)
        clin = causal_clinical_bandpass_360hz(afe, fs=360.0)

        sqi = extract_sqi_features(afe, clin, fs=360.0)
        feat = sqi["features"]
        orph_acc = bool(sqi["orphanidou_accept"])
        cliff_acc = bool(sqi["clifford_accept"])
        label = 1 if rec_id in unacc_set else 0

        # Continuous heuristic scores matching MIT-BIH definitions
        # Clifford: (f1 + f3 + f5)/3
        cliff_score = float((feat[0] + feat[2] + feat[4]) / 3.0)
        # Orphanidou: (f1 + (1 - single_corr))/2
        single_corr = float(sqi.get("single_corr", 0.0))
        orph_score = float((feat[0] + max(0.0, 1.0 - single_corr)) / 2.0)
        # Single MorphDisp: f2
        morph_score = float(feat[1])

        # Frozen composite score (derived from MIT-BIH fit set)
        score = float(score_uncertainty(feat[np.newaxis, :], med, iqr, w)[0])

        records.append({
            "rec_id": rec_id,
            "label": label,
            "features": feat,
            "score": score,
            "morph_score": morph_score,
            "cliff_score": cliff_score,
            "orph_score": orph_score,
            "orph_acc": orph_acc,
            "cliff_acc": cliff_acc,
        })

    print(f"Loaded and processed {len(records)} CinC 2011 records in {time.time() - t0:.1f}s")
    return records


def run_cinc_eval(
    data_dir: Path,
    results_dir: Path,
    tab_dir: Path,
) -> dict[str, Any]:
    cinc_dir = data_dir / "cinc2011"
    frozen_path = results_dir / "frozen_score_params.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"Missing {frozen_path}. Run run_benchmark.py first.")

    frozen_params = json.loads(frozen_path.read_text(encoding="utf-8"))
    records = load_cinc2011_records(cinc_dir, frozen_params)
    n = len(records)
    labels = np.array([r["label"] for r in records], dtype=int)
    scores = np.array([r["score"] for r in records], dtype=float)
    features = np.array([r["features"] for r in records], dtype=float)
    morph_scores = np.array([r["morph_score"] for r in records], dtype=float)
    cliff_scores = np.array([r["cliff_score"] for r in records], dtype=float)
    orph_scores = np.array([r["orph_score"] for r in records], dtype=float)
    orph_acc = np.array([r["orph_acc"] for r in records], dtype=bool)
    cliff_acc = np.array([r["cliff_acc"] for r in records], dtype=bool)

    n_fail = int(np.sum(labels))
    base_fail_rate = n_fail / n
    score_auroc = float(roc_auc_score(labels, scores))

    def eval_mask(mask: np.ndarray) -> dict[str, Any]:
        acc_cnt = int(np.sum(mask))
        fail_cnt = int(np.sum(mask * labels))
        sr_p, sr_lo, sr_hi = wilson_ci(fail_cnt, n)
        cf_p, cf_lo, cf_hi = wilson_ci(fail_cnt, acc_cnt) if acc_cnt > 0 else (0.0, 0.0, 0.0)
        return {
            "accept_count": acc_cnt,
            "accept_pct": (acc_cnt / n) * 100.0,
            "silent_fail_count": fail_cnt,
            "silent_risk_pct": sr_p * 100.0,
            "silent_risk_ci_lo_pct": sr_lo * 100.0,
            "silent_risk_ci_hi_pct": sr_hi * 100.0,
            "cond_fail_pct": cf_p * 100.0,
            "cond_fail_ci_lo_pct": cf_lo * 100.0,
            "cond_fail_ci_hi_pct": cf_hi * 100.0,
        }

    heuristics = {
        "all_accept": eval_mask(np.ones(n, dtype=bool)),
        "clifford": eval_mask(cliff_acc),
        "orphanidou": eval_mask(orph_acc),
    }

    # 1. PROTOCOL 1: Zero-Recalibration Transfer from MIT-BIH
    source_taus = frozen_params.get("tau_medians", {})
    zero_transfer = {}
    for a in ALPHAS:
        k = f"{a:.2f}"
        if k in source_taus:
            tau = float(source_taus[k])
            mask = scores <= tau
            zero_transfer[k] = eval_mask(mask)
            zero_transfer[k]["tau_source"] = tau

    # 2. PROTOCOL 2: Target-Recalibrated Conformal Risk Control (5-Fold CV on CinC 2011)
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)

    def run_target_cv_generic(score_fn: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cv_out = {f"{a:.2f}": {"accept_pct": [], "silent_risk_pct": [], "cond_fail_pct": [], "tau": []} for a in ALPHAS}
        oof_accept = {f"{a:.2f}": np.zeros(n, dtype=bool) for a in ALPHAS}

        for train_idx, test_idx in kf.split(records):
            out_sc = score_fn(train_idx, test_idx)
            if len(out_sc) == 3:
                cal_scores, test_scores, cal_idx = out_sc
            else:
                cal_scores, test_scores = out_sc
                cal_idx = train_idx
            cal_losses = labels[cal_idx]
            cal_pids = np.array([f"cinc_{i}" for i in cal_idx])
            test_losses = labels[test_idx]

            for a in ALPHAS:
                k = f"{a:.2f}"
                tau = calibrate_crc_threshold(cal_scores, cal_losses, cal_pids, alpha=a)
                test_mask = test_scores <= tau
                oof_accept[k][test_idx] = test_mask

                test_acc = float(np.mean(test_mask)) * 100.0
                test_sr = float(np.mean(test_mask * test_losses)) * 100.0
                test_cf = float(np.sum(test_mask * test_losses) / max(np.sum(test_mask), 1)) * 100.0

                cv_out[k]["accept_pct"].append(test_acc)
                cv_out[k]["silent_risk_pct"].append(test_sr)
                cv_out[k]["cond_fail_pct"].append(test_cf)
                cv_out[k]["tau"].append(tau)

        agg = {}
        for a in ALPHAS:
            k = f"{a:.2f}"
            agg[k] = {
                "accept_pct": float(np.mean(cv_out[k]["accept_pct"])),
                "accept_se": float(np.std(cv_out[k]["accept_pct"]) / np.sqrt(5)),
                "silent_risk_pct": float(np.mean(cv_out[k]["silent_risk_pct"])),
                "silent_risk_se": float(np.std(cv_out[k]["silent_risk_pct"]) / np.sqrt(5)),
                "cond_fail_pct": float(np.mean(cv_out[k]["cond_fail_pct"])),
                "cond_fail_se": float(np.std(cv_out[k]["cond_fail_pct"]) / np.sqrt(5)),
                "tau_mean": float(np.mean(cv_out[k]["tau"])),
            }
        return agg, oof_accept

    target_crc, oof_prop = run_target_cv_generic(lambda tr, te: (scores[tr], scores[te]))
    target_cliff_crc, oof_cliff = run_target_cv_generic(lambda tr, te: (cliff_scores[tr], cliff_scores[te]))
    target_orph_crc, oof_orph = run_target_cv_generic(lambda tr, te: (orph_scores[tr], orph_scores[te]))
    target_morph_crc, oof_morph = run_target_cv_generic(lambda tr, te: (morph_scores[tr], morph_scores[te]))

    def lr_score_fn(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm_tr = np.random.default_rng(2026 + int(tr[0])).permutation(tr)
        n_fit = max(20, int(round(0.30 * len(perm_tr))))
        fit_tr, cal_tr = perm_tr[:n_fit], perm_tr[n_fit:]
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=2026)
        clf.fit(features[fit_tr], labels[fit_tr])
        return clf.predict_proba(features[cal_tr])[:, 1], clf.predict_proba(features[te])[:, 1], cal_tr

    target_lr_crc, oof_lr = run_target_cv_generic(lr_score_fn)

    # Paired Bootstrap Inference on Acceptance Differences at alpha = 0.08
    rng_b = np.random.default_rng(2026)
    n_boot = 2000
    boot_diffs = {
        "cliff": [],
        "orph": [],
        "morph": [],
        "lr": [],
    }
    m_prop_08 = oof_prop["0.08"]
    m_cliff_08 = oof_cliff["0.08"]
    m_orph_08 = oof_orph["0.08"]
    m_morph_08 = oof_morph["0.08"]
    m_lr_08 = oof_lr["0.08"]

    for _ in range(n_boot):
        b_idx = rng_b.integers(0, n, size=n)
        acc_p = np.mean(m_prop_08[b_idx]) * 100.0
        boot_diffs["cliff"].append(acc_p - np.mean(m_cliff_08[b_idx]) * 100.0)
        boot_diffs["orph"].append(acc_p - np.mean(m_orph_08[b_idx]) * 100.0)
        boot_diffs["morph"].append(acc_p - np.mean(m_morph_08[b_idx]) * 100.0)
        boot_diffs["lr"].append(acc_p - np.mean(m_lr_08[b_idx]) * 100.0)

    paired_diffs = {}
    for key, arr in boot_diffs.items():
        arr = np.asarray(arr)
        paired_diffs[key] = {
            "mean_delta_pct": float(np.mean(arr)),
            "ci_lo_pct": float(np.percentile(arr, 2.5)),
            "ci_hi_pct": float(np.percentile(arr, 97.5)),
        }

    results = {
        "dataset": "PhysioNet / CinC Challenge 2011 Set A",
        "n_records": n,
        "n_unacceptable": n_fail,
        "baseline_unacceptable_pct": base_fail_rate * 100.0,
        "score_auroc": score_auroc,
        "heuristics": heuristics,
        "zero_recalibration_transfer": zero_transfer,
        "target_recalibrated_crc": target_crc,
        "target_clifford_crc": target_cliff_crc,
        "target_orphanidou_crc": target_orph_crc,
        "target_morph_crc": target_morph_crc,
        "target_lr_crc": target_lr_crc,
        "paired_acceptance_deltas_08": paired_diffs,
    }
    return results


def write_cinc_latex_table(res: dict[str, Any], tab_dir: Path) -> None:
    heur = res["heuristics"]
    zero = res["zero_recalibration_transfer"]
    targ = res["target_recalibrated_crc"]
    c_targ = res["target_clifford_crc"]
    o_targ = res["target_orphanidou_crc"]
    m_targ = res["target_morph_crc"]
    lr_targ = res["target_lr_crc"]
    diffs = res["paired_acceptance_deltas_08"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{External evaluation on real mobile-phone ECGs (PhysioNet/CinC Challenge 2011 Set A, $N={res['n_records']}$, unacceptable rate ${res['baseline_unacceptable_pct']:.1f}\\%$, unsupervised score AUROC ${res['score_auroc']:.3f}$). Paired $\\Delta$ denotes acceptance gain over baselines at $\\alpha=0.08$ with 95\\% bootstrap CI.}}",
        r"\label{tab:cinc2011}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method / Gating Policy & Nominal $\alpha$ & Accept (\%) & Silent Risk (\%) & Cond.\ Fail (\%) & $\Delta$ Accept vs.\ Unsup.\ (\%) \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Literature Heuristics (Fixed Operating Points)}} \\",
        f"All-Accept Baseline & --- & {heur['all_accept']['accept_pct']:.1f} & {heur['all_accept']['silent_risk_pct']:.1f} [{heur['all_accept']['silent_risk_ci_lo_pct']:.1f}, {heur['all_accept']['silent_risk_ci_hi_pct']:.1f}] & {heur['all_accept']['cond_fail_pct']:.1f} & --- \\\\",
        f"Clifford et al. (2012) & --- & {heur['clifford']['accept_pct']:.1f} & {heur['clifford']['silent_risk_pct']:.1f} [{heur['clifford']['silent_risk_ci_lo_pct']:.1f}, {heur['clifford']['silent_risk_ci_hi_pct']:.1f}] & {heur['clifford']['cond_fail_pct']:.1f} & --- \\\\",
        f"Orphanidou et al. (2015) & --- & {heur['orphanidou']['accept_pct']:.1f} & {heur['orphanidou']['silent_risk_pct']:.1f} [{heur['orphanidou']['silent_risk_ci_lo_pct']:.1f}, {heur['orphanidou']['silent_risk_ci_hi_pct']:.1f}] & {heur['orphanidou']['cond_fail_pct']:.1f} & --- \\\\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Protocol 1: Zero-Recalibration Transfer from MIT-BIH (Zero-Shot)}} \\",
    ]

    for a_str in ("0.05", "0.08", "0.10"):
        if a_str in zero:
            z = zero[a_str]
            lines.append(
                f"Unsup. Composite Gate & {float(a_str):.2f} & {z['accept_pct']:.1f} & "
                f"{z['silent_risk_pct']:.2f} [{z['silent_risk_ci_lo_pct']:.2f}, {z['silent_risk_ci_hi_pct']:.2f}] & "
                f"{z['cond_fail_pct']:.1f} & --- \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Protocol 2: Target-Recalibrated Conformal Risk Control (5-Fold CV)}} \\",
    ])

    for a_str in ("0.05", "0.08", "0.10"):
        t = targ[a_str]
        lines.append(
            fr"Unsup. Composite CRC & {float(a_str):.2f} & {t['accept_pct']:.1f}$\pm${t['accept_se']:.1f} & "
            fr"\textbf{{{t['silent_risk_pct']:.2f}}}$\pm${t['silent_risk_se']:.2f} & "
            fr"{t['cond_fail_pct']:.1f}$\pm${t['cond_fail_se']:.1f} & --- \\"
        )

    # Baselines at matched alpha = 0.08 with paired delta
    d_c = diffs["cliff"]
    lines.append(
        fr"Clifford (CRC) & 0.08 & {c_targ['0.08']['accept_pct']:.1f}$\pm${c_targ['0.08']['accept_se']:.1f} & "
        fr"{c_targ['0.08']['silent_risk_pct']:.2f}$\pm${c_targ['0.08']['silent_risk_se']:.2f} & "
        fr"{c_targ['0.08']['cond_fail_pct']:.1f}$\pm${c_targ['0.08']['cond_fail_se']:.1f} & "
        fr"{d_c['mean_delta_pct']:+.1f} [{d_c['ci_lo_pct']:.1f}, {d_c['ci_hi_pct']:.1f}] \\"
    )

    d_o = diffs["orph"]
    lines.append(
        fr"Orphanidou (CRC) & 0.08 & {o_targ['0.08']['accept_pct']:.1f}$\pm${o_targ['0.08']['accept_se']:.1f} & "
        fr"{o_targ['0.08']['silent_risk_pct']:.2f}$\pm${o_targ['0.08']['silent_risk_se']:.2f} & "
        fr"{o_targ['0.08']['cond_fail_pct']:.1f}$\pm${o_targ['0.08']['cond_fail_se']:.1f} & "
        fr"{d_o['mean_delta_pct']:+.1f} [{d_o['ci_lo_pct']:.1f}, {d_o['ci_hi_pct']:.1f}] \\"
    )

    d_m = diffs["morph"]
    lines.append(
        fr"Single MorphDisp (CRC) & 0.08 & {m_targ['0.08']['accept_pct']:.1f}$\pm${m_targ['0.08']['accept_se']:.1f} & "
        fr"{m_targ['0.08']['silent_risk_pct']:.2f}$\pm${m_targ['0.08']['silent_risk_se']:.2f} & "
        fr"{m_targ['0.08']['cond_fail_pct']:.1f}$\pm${m_targ['0.08']['cond_fail_se']:.1f} & "
        fr"{d_m['mean_delta_pct']:+.1f} [{d_m['ci_lo_pct']:.1f}, {d_m['ci_hi_pct']:.1f}] \\"
    )

    d_lr = diffs["lr"]
    lines.append(
        fr"Logistic Regression (CRC) & 0.08 & {lr_targ['0.08']['accept_pct']:.1f}$\pm${lr_targ['0.08']['accept_se']:.1f} & "
        fr"{lr_targ['0.08']['silent_risk_pct']:.2f}$\pm${lr_targ['0.08']['silent_risk_se']:.2f} & "
        fr"{lr_targ['0.08']['cond_fail_pct']:.1f}$\pm${lr_targ['0.08']['cond_fail_se']:.1f} & "
        fr"{d_lr['mean_delta_pct']:+.1f} [{d_lr['ci_lo_pct']:.1f}, {d_lr['ci_hi_pct']:.1f}] \\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tab_dir / "tab_cinc2011.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote CinC 2011 LaTeX table to {tab_dir / 'tab_cinc2011.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on PhysioNet CinC 2011 Set A.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Path to data directory")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Path to results directory")
    parser.add_argument("--tab-dir", type=Path, default=ROOT / "results" / "latex" / "tables", help="Path to tables directory")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.tab_dir.mkdir(parents=True, exist_ok=True)

    res = run_cinc_eval(args.data_dir, args.results_dir, args.tab_dir)
    (args.results_dir / "cinc2011_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_cinc_latex_table(res, args.tab_dir)


if __name__ == "__main__":
    main()
