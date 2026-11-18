#!/usr/bin/env python3
"""External wearable evaluation on PhysioNet/CinC Challenge 2017 AliveCor single-lead ECGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ecgrisk.afe import apply_diagnostic_afe_360hz, causal_clinical_bandpass_360hz
from ecgrisk.conformal import calibrate_crc_threshold, score_uncertainty, wilson_binomial_ci
from ecgrisk.sqi import extract_sqi_features

ALPHAS_17 = [0.005, 0.01, 0.015, 0.02]


def load_cinc2017_records(
    cinc17_dir: Path,
    frozen_params: dict[str, Any],
) -> list[dict[str, Any]]:
    ref_path = cinc17_dir / "REFERENCE.csv"
    train_dir = cinc17_dir / "training2017"
    if not ref_path.exists():
        ref_path = train_dir / "REFERENCE.csv"
    if not ref_path.exists() or not train_dir.exists():
        raise FileNotFoundError(
            f"Missing CinC 2017 data in {cinc17_dir}. "
            "Run `python3 experiments/download_data.py --datasets cinc2017` first."
        )

    med = np.asarray(frozen_params["medians"], dtype=float)
    iqr = np.asarray(frozen_params["iqrs"], dtype=float)
    w = np.asarray(frozen_params["weights"], dtype=float)

    rows = []
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        rec_id, cls_lbl = [p.strip() for p in line.split(",", 1)]
        rec_name = rec_id.split("/")[-1]
        mat_path = train_dir / f"{rec_name}.mat"
        if not mat_path.exists():
            continue
        rows.append((rec_name, cls_lbl, mat_path))

    records: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, (rec_name, cls_lbl, mat_path) in enumerate(rows):
        mat = loadmat(str(mat_path))
        sig_300 = np.asarray(mat["val"], dtype=float).ravel() / 1000.0  # 1000 ADC/mV -> mV
        if np.any(np.isnan(sig_300)):
            sig_300 = np.nan_to_num(sig_300, nan=0.0)
        # Resample 300 Hz -> 360 Hz (6/5)
        sig_360 = resample_poly(sig_300, 6, 5)
        # Standardize evaluation on up to first 10 s (3600 samples) matching telemetry packet size
        sig_360 = sig_360[: min(len(sig_360), 3600)]

        afe = apply_diagnostic_afe_360hz(sig_360, fs=360.0)
        clin = causal_clinical_bandpass_360hz(afe, fs=360.0)
        sqi = extract_sqi_features(afe, clin, fs=360.0)
        feat = sqi["features"]

        single_corr = float(sqi.get("single_corr", 0.0))
        cliff_score = float((feat[0] + feat[2] + feat[4]) / 3.0)
        orph_score = float((feat[0] + max(0.0, 1.0 - single_corr)) / 2.0)
        morph_score = float(feat[1])
        score = float(score_uncertainty(feat[np.newaxis, :], med, iqr, w)[0])

        records.append(
            {
                "rec_id": rec_name,
                "class": cls_lbl,
                "label": 1 if cls_lbl == "~" else 0,
                "features": feat,
                "score": score,
                "morph_score": morph_score,
                "cliff_score": cliff_score,
                "orph_score": orph_score,
                "orph_acc": bool(sqi["orphanidou_accept"]),
                "cliff_acc": bool(sqi["clifford_accept"]),
            }
        )
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} CinC 2017 records ({time.time() - t0:.1f}s)")

    print(f"Loaded and processed {len(records)} CinC 2017 records in {time.time() - t0:.1f}s")
    return records


def run_cinc17_eval(
    data_dir: Path,
    results_dir: Path,
    tab_dir: Path,
) -> dict[str, Any]:
    cinc17_dir = data_dir / "cinc2017"
    frozen_path = results_dir / "frozen_score_params.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"Missing {frozen_path}. Run run_benchmark.py first.")

    frozen_params = json.loads(frozen_path.read_text(encoding="utf-8"))
    records = load_cinc2017_records(cinc17_dir, frozen_params)
    n = len(records)
    labels = np.array([r["label"] for r in records], dtype=int)
    classes = np.array([r["class"] for r in records])
    is_afib = classes == "A"
    is_norm = classes == "N"

    scores = np.array([r["score"] for r in records], dtype=float)
    features = np.array([r["features"] for r in records], dtype=float)
    morph_scores = np.array([r["morph_score"] for r in records], dtype=float)
    cliff_scores = np.array([r["cliff_score"] for r in records], dtype=float)
    orph_scores = np.array([r["orph_score"] for r in records], dtype=float)
    orph_acc = np.array([r["orph_acc"] for r in records], dtype=bool)
    cliff_acc = np.array([r["cliff_acc"] for r in records], dtype=bool)

    n_noisy = int(np.sum(labels))
    base_noisy_pct = (n_noisy / n) * 100.0
    score_auroc = float(roc_auc_score(labels, scores))

    def eval_mask(mask: np.ndarray) -> dict[str, Any]:
        acc_cnt = int(np.sum(mask))
        fail_cnt = int(np.sum(mask * labels))
        lo, hi = wilson_binomial_ci(fail_cnt, n)
        return {
            "accept_pct": (acc_cnt / n) * 100.0,
            "silent_risk_pct": (fail_cnt / n) * 100.0,
            "silent_risk_ci_lo_pct": lo * 100.0,
            "silent_risk_ci_hi_pct": hi * 100.0,
            "cond_fail_pct": (fail_cnt / max(acc_cnt, 1)) * 100.0,
            "afib_accept_pct": float(np.mean(mask[is_afib])) * 100.0 if np.any(is_afib) else np.nan,
            "normal_accept_pct": float(np.mean(mask[is_norm])) * 100.0 if np.any(is_norm) else np.nan,
        }

    heuristics = {
        "all_accept": eval_mask(np.ones(n, dtype=bool)),
        "clifford": eval_mask(cliff_acc),
        "orphanidou": eval_mask(orph_acc),
    }

    source_taus = frozen_params.get("tau_medians", {})
    zero_transfer = {}
    if "0.08" in source_taus:
        tau_src = float(source_taus["0.08"])
        zero_transfer["0.08"] = eval_mask(scores <= tau_src)
        zero_transfer["0.08"]["tau_source"] = tau_src

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

    def run_target_cv(score_fn: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cv_out = {
            f"{a:.3f}": {"accept_pct": [], "silent_risk_pct": [], "cond_fail_pct": [], "afib_accept_pct": [], "normal_accept_pct": []}
            for a in ALPHAS_17
        }
        oof_accept = {f"{a:.3f}": np.zeros(n, dtype=bool) for a in ALPHAS_17}

        for train_idx, test_idx in skf.split(features, labels):
            out_sc = score_fn(train_idx, test_idx)
            if len(out_sc) == 3:
                cal_sc, test_sc, cal_idx = out_sc
            else:
                cal_sc, test_sc = out_sc
                cal_idx = train_idx
            cal_losses = labels[cal_idx]
            cal_pids = np.array([f"c17_{i}" for i in cal_idx])
            test_losses = labels[test_idx]
            te_afib = is_afib[test_idx]
            te_norm = is_norm[test_idx]

            for a in ALPHAS_17:
                k = f"{a:.3f}"
                tau = calibrate_crc_threshold(cal_sc, cal_losses, cal_pids, alpha=a)
                t_mask = test_sc <= tau
                oof_accept[k][test_idx] = t_mask
                cv_out[k]["accept_pct"].append(float(np.mean(t_mask)) * 100.0)
                cv_out[k]["silent_risk_pct"].append(float(np.mean(t_mask * test_losses)) * 100.0)
                cv_out[k]["cond_fail_pct"].append(float(np.sum(t_mask * test_losses) / max(np.sum(t_mask), 1)) * 100.0)
                cv_out[k]["afib_accept_pct"].append(float(np.mean(t_mask[te_afib])) * 100.0)
                cv_out[k]["normal_accept_pct"].append(float(np.mean(t_mask[te_norm])) * 100.0)

        agg = {}
        for a in ALPHAS_17:
            k = f"{a:.3f}"
            agg[k] = {
                "accept_pct": float(np.mean(cv_out[k]["accept_pct"])),
                "accept_se": float(np.std(cv_out[k]["accept_pct"]) / np.sqrt(5)),
                "silent_risk_pct": float(np.mean(cv_out[k]["silent_risk_pct"])),
                "silent_risk_se": float(np.std(cv_out[k]["silent_risk_pct"]) / np.sqrt(5)),
                "cond_fail_pct": float(np.mean(cv_out[k]["cond_fail_pct"])),
                "afib_accept_pct": float(np.mean(cv_out[k]["afib_accept_pct"])),
                "normal_accept_pct": float(np.mean(cv_out[k]["normal_accept_pct"])),
            }
        return agg, oof_accept

    targ_unsup, oof_unsup = run_target_cv(lambda tr, te: (scores[tr], scores[te]))
    targ_cliff, oof_cliff = run_target_cv(lambda tr, te: (cliff_scores[tr], cliff_scores[te]))
    targ_orph, oof_orph = run_target_cv(lambda tr, te: (orph_scores[tr], orph_scores[te]))
    targ_morph, oof_morph = run_target_cv(lambda tr, te: (morph_scores[tr], morph_scores[te]))

    def lr_fn(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perm_tr = np.random.default_rng(2026 + int(tr[0])).permutation(tr)
        n_fit = max(50, int(round(0.30 * len(perm_tr))))
        fit_tr, cal_tr = perm_tr[:n_fit], perm_tr[n_fit:]
        clf = LogisticRegression(C=1.0, max_iter=300, random_state=2026)
        clf.fit(features[fit_tr], labels[fit_tr])
        return clf.predict_proba(features[cal_tr])[:, 1], clf.predict_proba(features[te])[:, 1], cal_tr

    targ_lr, oof_lr = run_target_cv(lr_fn)

    # Paired bootstrap on AFib preservation delta and overall acceptance delta at alpha = 0.015
    rng_b = np.random.default_rng(2026)
    n_boot = 2000
    afib_idx = np.where(is_afib)[0]
    boot_acc = {"cliff": [], "orph": [], "morph": [], "lr": []}
    boot_afib = {"cliff": [], "orph": [], "morph": [], "lr": []}
    key_match = "0.015"
    for _ in range(n_boot):
        bi = rng_b.integers(0, n, size=n)
        ba = rng_b.choice(afib_idx, size=len(afib_idx), replace=True)
        p_a = float(np.mean(oof_unsup[key_match][bi])) * 100.0
        p_af = float(np.mean(oof_unsup[key_match][ba])) * 100.0
        for name, oof_d in (("cliff", oof_cliff), ("orph", oof_orph), ("morph", oof_morph), ("lr", oof_lr)):
            boot_acc[name].append(p_a - float(np.mean(oof_d[key_match][bi])) * 100.0)
            boot_afib[name].append(p_af - float(np.mean(oof_d[key_match][ba])) * 100.0)

    paired_deltas = {}
    for name in ("cliff", "orph", "morph", "lr"):
        arr_a = np.asarray(boot_acc[name])
        arr_f = np.asarray(boot_afib[name])
        paired_deltas[name] = {
            "accept_delta_pct": float(np.mean(arr_a)),
            "accept_ci_lo": float(np.percentile(arr_a, 2.5)),
            "accept_ci_hi": float(np.percentile(arr_a, 97.5)),
            "afib_delta_pct": float(np.mean(arr_f)),
            "afib_ci_lo": float(np.percentile(arr_f, 2.5)),
            "afib_ci_hi": float(np.percentile(arr_f, 97.5)),
        }

    return {
        "dataset": "PhysioNet / CinC Challenge 2017 (AliveCor Single-Lead)",
        "n_records": n,
        "n_noisy": n_noisy,
        "n_afib": int(np.sum(is_afib)),
        "n_normal": int(np.sum(is_norm)),
        "baseline_noisy_pct": base_noisy_pct,
        "score_auroc": score_auroc,
        "heuristics": heuristics,
        "zero_recalibration_transfer": zero_transfer,
        "target_unsup_crc": targ_unsup,
        "target_clifford_crc": targ_cliff,
        "target_orphanidou_crc": targ_orph,
        "target_morph_crc": targ_morph,
        "target_lr_crc": targ_lr,
        "paired_deltas_015": paired_deltas,
    }


def write_cinc17_latex_table(res: dict[str, Any], tab_dir: Path) -> None:
    heur = res["heuristics"]
    targ = res["target_unsup_crc"]
    c_t = res["target_clifford_crc"]["0.015"]
    o_t = res["target_orphanidou_crc"]["0.015"]
    m_t = res["target_morph_crc"]["0.015"]
    lr_t = res["target_lr_crc"]["0.015"]
    diffs = res["paired_deltas_015"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            f"\\caption{{External wearable single-lead evaluation on PhysioNet/CinC 2017 AliveCor ECGs "
            f"($N={res['n_records']:,}$, noisy rate ${res['baseline_noisy_pct']:.2f}\\%$, AFib $N={res['n_afib']}$, "
            f"unsupervised AUROC ${res['score_auroc']:.3f}$). Paired $\\Delta$ AFib denotes AFib preservation gain at $\\alpha=0.015$.}}"
        ),
        r"\label{tab:cinc2017}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method / Gating Policy & $\alpha$ & Accept (\%) & Silent Risk (\%) & AFib Acc.\ (\%) & $\Delta$ AFib vs.\ Unsup.\ (\%) \\",
        r"\midrule",
        f"All-Accept Baseline & --- & 100.0 & {heur['all_accept']['silent_risk_pct']:.2f} & 100.0 & --- \\\\",
        f"Clifford et al. (2012) & --- & {heur['clifford']['accept_pct']:.1f} & {heur['clifford']['silent_risk_pct']:.2f} & {heur['clifford']['afib_accept_pct']:.1f} & --- \\\\",
        f"Orphanidou et al. (2015) & --- & {heur['orphanidou']['accept_pct']:.1f} & {heur['orphanidou']['silent_risk_pct']:.2f} & {heur['orphanidou']['afib_accept_pct']:.1f} & --- \\\\",
        r"\midrule",
    ]
    for a_str in ("0.010", "0.015", "0.020"):
        t = targ[a_str]
        lines.append(
            fr"Unsup. Composite CRC & {float(a_str):.3f} & {t['accept_pct']:.1f}$\pm${t['accept_se']:.1f} & "
            fr"\textbf{{{t['silent_risk_pct']:.2f}}}$\pm${t['silent_risk_se']:.2f} & "
            fr"{t['afib_accept_pct']:.1f} & --- \\"
        )
    for label, row, dk in (
        ("Clifford (CRC)", c_t, "cliff"),
        ("Orphanidou (CRC)", o_t, "orph"),
        ("Single MorphDisp (CRC)", m_t, "morph"),
        ("Logistic Regression (CRC)", lr_t, "lr"),
    ):
        d = diffs[dk]
        lines.append(
            fr"{label} & 0.015 & {row['accept_pct']:.1f}$\pm${row['accept_se']:.1f} & "
            fr"{row['silent_risk_pct']:.2f}$\pm${row['silent_risk_se']:.2f} & "
            fr"{row['afib_accept_pct']:.1f} & "
            fr"{d['afib_delta_pct']:+.1f} [{d['afib_ci_lo']:.1f}, {d['afib_ci_hi']:.1f}] \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tab_dir / "tab_cinc2017.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote CinC 2017 LaTeX table to {tab_dir / 'tab_cinc2017.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on PhysioNet CinC 2017 AliveCor ECGs.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tab-dir", type=Path, default=ROOT / "results" / "latex" / "tables")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.tab_dir.mkdir(parents=True, exist_ok=True)
    res = run_cinc17_eval(args.data_dir, args.results_dir, args.tab_dir)
    (args.results_dir / "cinc2017_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_cinc17_latex_table(res, args.tab_dir)


if __name__ == "__main__":
    main()
