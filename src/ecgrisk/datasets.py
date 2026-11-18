"""MIT-BIH Arrhythmia and NSTDB loaders at 360 Hz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import wfdb
except ImportError:
    wfdb = None

MITDB_RECORDS_48: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)

PACED_RECORDS: set[str] = {"102", "104", "107", "217"}
BEAT_SYMBOLS: set[str] = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
}
ECTOPIC_SYMBOLS: set[str] = {"V", "A", "a", "J", "S", "r", "F", "E", "e", "j"}


def classify_patient_group(pid: str, symbols: list[str]) -> str:
    """Classify patient rhythm as Paced, Ectopic (>=30 or >=3%), or Sinus."""
    if pid in PACED_RECORDS:
        return "Paced"
    if not symbols:
        return "Sinus"
    n_ect = sum(1 for s in symbols if s in ECTOPIC_SYMBOLS)
    if n_ect >= 30 or (n_ect / max(len(symbols), 1)) >= 0.03:
        return "Ectopic"
    return "Sinus"


def _select_primary_lead(sig_names: list[str]) -> int:
    for preferred in ("MLII", "V5", "V2", "V1", "ECG1"):
        for idx, name in enumerate(sig_names):
            if name.upper() == preferred:
                return idx
    return 0


def load_mitdb_contiguous_streams(
    mitdb_dir: str | Path,
    records: tuple[str, ...] = MITDB_RECORDS_48,
    n_strips_per_record: int = 60,
    strip_s: float = 10.0,
    start_s: float = 60.0,
    merge_201_202: bool = True,
    exclude_paced: bool = False,
) -> list[dict[str, Any]]:
    """Load contiguous 10 s strips at 360 Hz (merging records 201 and 202 into one patient by default)."""
    if wfdb is None:
        raise ImportError("wfdb is required (`pip install wfdb`).")
    mitdb_path = Path(mitdb_dir)
    if not mitdb_path.exists():
        raise FileNotFoundError(f"MIT-BIH directory not found: {mitdb_path}. Run `make download-data` to fetch it.")

    rows: list[dict[str, Any]] = []
    for pid in records:
        subject_id = "201" if (merge_201_202 and pid in ("201", "202")) else pid
        if exclude_paced and subject_id in PACED_RECORDS:
            continue

        rec = wfdb.rdrecord(str(mitdb_path / pid))
        ann = wfdb.rdann(str(mitdb_path / pid), "atr")
        fs = float(rec.fs)
        lead_idx = _select_primary_lead(list(rec.sig_name))
        raw_sig = rec.p_signal[:, lead_idx].astype(float)

        beat_mask = np.isin(ann.symbol, list(BEAT_SYMBOLS))
        beat_samples = np.asarray(ann.sample[beat_mask], dtype=int)
        beat_syms = [s for s, m in zip(ann.symbol, beat_mask) if m]
        group = classify_patient_group(subject_id, beat_syms)

        n_win = int(round(strip_s * fs))
        start_idx0 = min(int(round(start_s * fs)), max(0, len(raw_sig) - n_strips_per_record * n_win))
        offset_w = n_strips_per_record if (merge_201_202 and pid == "202") else 0

        for w_idx in range(n_strips_per_record):
            lo = start_idx0 + w_idx * n_win
            hi = lo + n_win
            if hi > len(raw_sig):
                break
            seg = raw_sig[lo:hi].copy()
            in_win = (beat_samples >= lo) & (beat_samples < hi)
            r_samples = beat_samples[in_win] - lo
            r_syms = [s for s, m in zip(beat_syms, in_win) if m]
            if len(r_samples) < 2:
                continue
            ect_flags = np.asarray([s in ECTOPIC_SYMBOLS for s in r_syms], dtype=bool)
            rows.append(
                {
                    "pid": subject_id,
                    "rec_id": pid,
                    "group": group,
                    "strip_idx": int(offset_w + w_idx),
                    "fs": fs,
                    "sig_clean_mv": seg,
                    "r_peaks_s": r_samples / fs,
                    "r_syms": r_syms,
                    "ect_flags": ect_flags,
                    "has_ectopic": bool(np.any(ect_flags)),
                }
            )
    return rows


def load_nstdb_360hz(nstdb_dir: str | Path, split: str = "all") -> dict[str, np.ndarray]:
    """Load NSTDB recordings (bw, em, ma) for split in {'all', 'cal', 'test'}."""
    if wfdb is None:
        raise ImportError("wfdb is required (`pip install wfdb`).")
    nstdb_path = Path(nstdb_dir)
    if not nstdb_path.exists():
        raise FileNotFoundError(f"NSTDB directory not found: {nstdb_path}. Run `make download-data` to fetch it.")
    out: dict[str, np.ndarray] = {}
    for key in ("bw", "em", "ma"):
        rec = wfdb.rdrecord(str(nstdb_path / key))
        noise = rec.p_signal[:, 0].astype(float)
        noise = noise - np.mean(noise)
        if split == "cal":
            noise = noise[:325000]
        elif split == "test":
            noise = noise[325000:]
        out[key] = noise / max(float(np.std(noise)), 1e-6)
    return out


def _scale_noise_to_snr(sig: np.ndarray, unit_noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_ac = sig - np.median(sig)
    sig_rms = max(float(np.std(sig_ac)), 0.05)
    target_noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
    return unit_noise * target_noise_rms


def generate_ambulatory_corpus(
    streams: list[dict[str, Any]],
    nstdb: dict[str, np.ndarray],
    seed: int = 2026,
    noise_regimes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Inject Markov-episodic NSTDB noise (-2 to 12 dB) with heterogeneous patient duties."""
    rng = np.random.default_rng(seed)
    by_pid: dict[str, list[dict[str, Any]]] = {}
    for row in streams:
        by_pid.setdefault(row["pid"], []).append(row)

    out_rows: list[dict[str, Any]] = []
    for pid in sorted(by_pid):
        p_rows = sorted(by_pid[pid], key=lambda r: r["strip_idx"])
        duty_p = float(rng.uniform(0.10, 0.65))
        p_stay_noisy = 0.72
        p_clean_to_noisy = max(0.02, min(0.50, duty_p * (1.0 - p_stay_noisy) / max(1.0 - duty_p, 1e-3)))
        noisy_state = bool(rng.random() < duty_p)

        for r in p_rows:
            sig_clean = r["sig_clean_mv"]
            n_samp = len(sig_clean)
            if noisy_state:
                if noise_regimes is not None:
                    regime = str(rng.choice(list(noise_regimes)))
                else:
                    regime = str(rng.choice(["em", "ma", "em+ma", "bw"], p=[0.40, 0.30, 0.20, 0.10]))
                snr_db = float(rng.uniform(-2.0, 12.0))
                if regime == "em+ma":
                    i0 = int(rng.integers(0, len(nstdb["em"]) - n_samp - 1))
                    i1 = int(rng.integers(0, len(nstdb["ma"]) - n_samp - 1))
                    unit_n = 0.7 * nstdb["em"][i0 : i0 + n_samp] + 0.7 * nstdb["ma"][i1 : i1 + n_samp]
                else:
                    src = nstdb[regime]
                    i0 = int(rng.integers(0, len(src) - n_samp - 1))
                    unit_n = src[i0 : i0 + n_samp]
                unit_n = unit_n / max(float(np.std(unit_n)), 1e-6)
                if rng.random() < 0.45:
                    burst_len = int(rng.integers(int(3.0 * r["fs"]), n_samp + 1))
                    b0 = int(rng.integers(0, max(1, n_samp - burst_len + 1)))
                    env = np.zeros(n_samp, dtype=float)
                    env[b0 : b0 + burst_len] = 1.0
                    added = _scale_noise_to_snr(sig_clean, unit_n, snr_db) * env
                else:
                    added = _scale_noise_to_snr(sig_clean, unit_n, snr_db)
                sig_obs = sig_clean + added
            else:
                regime = "clean"
                snr_db = np.inf
                sig_obs = sig_clean.copy()

            row_copy = dict(r)
            row_copy["sig_obs_mv"] = sig_obs
            row_copy["noise_regime"] = regime
            row_copy["snr_db"] = snr_db
            row_copy["patient_duty"] = duty_p
            out_rows.append(row_copy)

            noisy_state = bool(rng.random() < (p_stay_noisy if noisy_state else p_clean_to_noisy))
    return out_rows


def inject_stress_noise(
    streams: list[dict[str, Any]],
    nstdb: dict[str, np.ndarray],
    noise_key: str,
    snr_db: float,
    seed: int = 999,
) -> list[dict[str, Any]]:
    """Inject a fixed NSTDB noise type and SNR across all strips."""
    rng = np.random.default_rng(seed)
    src = nstdb[noise_key]
    out: list[dict[str, Any]] = []
    for r in streams:
        sig_clean = r["sig_clean_mv"]
        n_samp = len(sig_clean)
        i0 = int(rng.integers(0, len(src) - n_samp - 1))
        seg_n = src[i0 : i0 + n_samp]
        unit_n = seg_n / max(float(np.std(seg_n)), 1e-6)
        added = _scale_noise_to_snr(sig_clean, unit_n, snr_db)
        row_copy = dict(r)
        row_copy["sig_obs_mv"] = sig_clean + added
        row_copy["noise_regime"] = f"{noise_key}_{snr_db:.0f}dB"
        row_copy["snr_db"] = snr_db
        out.append(row_copy)
    return out
