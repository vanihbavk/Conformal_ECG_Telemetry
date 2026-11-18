"""Sanity checks on the committed results/ artifacts. No datasets needed."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ecgrisk.config import ConformalConfig
from ecgrisk.datasets import MITDB_RECORDS_48

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

ALPHA_KEYS = ("0.05", "0.06", "0.08", "0.10", "0.12", "0.15")
EXPECTED_FIGURES = (
    "system_block_diagram.pdf",
    "sqi_signals_examples.pdf",
    "cohort_and_nstdb_shift.pdf",
    "crc_validity_pareto.pdf",
)


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise unittest.SkipTest(f"{name} has not been generated yet.")
    return json.loads(path.read_text(encoding="utf-8"))


class TestResultsAudit(unittest.TestCase):
    def test_benchmark_design_matches_library(self) -> None:
        s = _load("summary.json")
        # records 201 and 202 are one subject
        self.assertEqual(s["n_patients"], len(MITDB_RECORDS_48) - 1)
        self.assertEqual(s["n_runs"], s["n_noise_seeds"] * s["splits_per_noise"])
        self.assertEqual(s["m_train"], s["m_cal"])
        self.assertEqual(s["m_fit"] + s["m_cal_sup"], s["m_train"])
        self.assertAlmostEqual(s["cal_slack_pct"], 100.0 / (s["m_cal"] + 1), places=6)
        self.assertAlmostEqual(s["cal_slack_sup_pct"], 100.0 / (s["m_cal_sup"] + 1), places=6)

    def test_frozen_params_match_config_and_summary(self) -> None:
        s = _load("summary.json")
        frozen = _load("frozen_score_params.json")
        cfg = ConformalConfig()
        self.assertEqual(frozen["iqrs"], list(cfg.fixed_unsup_scales))
        self.assertEqual(frozen["weights"], list(cfg.fixed_unsup_weights))
        self.assertEqual(frozen["tau_medians"], s["tau_medians"])
        self.assertAlmostEqual(frozen["tau_medians"]["0.08"], s["tau_08_median"], places=12)
        self.assertAlmostEqual(sum(frozen["weights"]), 1.0, places=9)

    def test_thresholds_are_monotone_in_alpha(self) -> None:
        taus = [_load("summary.json")["tau_medians"][a] for a in ALPHA_KEYS]
        self.assertEqual(taus, sorted(taus))

    def test_crc_controls_risk_at_nominal_alpha(self) -> None:
        crc = _load("summary.json")["cv_nominal"]["crc"]
        for a in ALPHA_KEYS:
            row = crc[a]
            self.assertLessEqual(row["patient_risk_pct"], 100.0 * float(a), f"risk above alpha at {a}")
            self.assertLessEqual(row["patient_risk_ci_lo_pct"], row["patient_risk_pct"])
            self.assertLessEqual(row["patient_risk_pct"], row["patient_risk_ci_hi_pct"])
            self.assertAlmostEqual(row["accept_pct"] + row["abstain_pct"], 100.0, places=6)

    def test_acceptance_increases_with_alpha(self) -> None:
        crc = _load("summary.json")["cv_nominal"]["crc"]
        acc = [crc[a]["accept_pct"] for a in ALPHA_KEYS]
        self.assertEqual(acc, sorted(acc))

    def test_external_cohort_sizes(self) -> None:
        c11 = _load("cinc2011_summary.json")
        c17 = _load("cinc2017_summary.json")
        self.assertEqual(c11["n_records"], 998)
        self.assertEqual(c17["n_records"], 8528)
        self.assertEqual(c17["n_noisy"] + c17["n_afib"] + c17["n_normal"] <= c17["n_records"], True)
        for res in (c11, c17):
            self.assertGreater(res["score_auroc"], 0.5)
            self.assertLessEqual(res["score_auroc"], 1.0)

    def test_unseen_noise_results_flag_matches_risk(self) -> None:
        u = _load("unseen_noise_results.json")
        for name, row in u["results"].items():
            self.assertEqual(row["exceeds_alpha_08"], row["risk_pct"] > 8.0, name)

    def test_figures_present(self) -> None:
        for name in EXPECTED_FIGURES:
            path = FIGURES / name
            self.assertTrue(path.exists(), f"missing results/figures/{name}")
            self.assertTrue(path.read_bytes().startswith(b"%PDF"), f"{name} is not a PDF")


if __name__ == "__main__":
    unittest.main()
