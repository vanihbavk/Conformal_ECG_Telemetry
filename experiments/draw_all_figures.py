"""Generate publication figures (both grayscale and muted color versions, 6.5 in text width) from results/ and data/."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DEFAULT_FIG_DIR = RESULTS_DIR / "figures"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ecgrisk.afe import apply_diagnostic_afe_360hz, causal_clinical_bandpass_360hz
from ecgrisk.conformal import score_uncertainty
from ecgrisk.datasets import (
    generate_ambulatory_corpus,
    load_mitdb_contiguous_streams,
    load_nstdb_360hz,
)
from ecgrisk.sqi import extract_sqi_features

WIDTH = 6.5  # inches, matches the paper text width

BOLD = {"fontfamily": "cmb10"}
ITALIC = {"fontfamily": "cmti10"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e2e5e8",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "legend.frameon": False,
})

summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
frozen = json.loads((RESULTS_DIR / "frozen_score_params.json").read_text(encoding="utf-8"))
cv = summary["cv_nominal"]
heur = summary["heuristics"]
cohort = summary["cohort_subgroups"]
stress = summary["stress_nstdb"]
ALPHAS = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
AKEYS = [f"{a:.2f}" for a in ALPHAS]
M_CAL = summary["m_cal"]
M_SUP = summary["m_cal_sup"]


def get_palette(color: bool = False) -> dict[str, str]:
    if color:
        return {
            "navy": "#2c4f7c",
            "navy_border": "#1e3757",
            "terracotta": "#a64b48",
            "terracotta_border": "#753330",
            "ochre": "#b87333",
            "ochre_border": "#7a4c21",
            "slate": "#5f6c7b",
            "slate_border": "#3b444e",
            "green": "#3b7a57",
            "pale_ci": "#d2e2ee",
            "abst_bar": "#9bb0c1",
            "abst_border": "#506b80",
            "viol_bar": "#c97e7b",
            "viol_border": "#8a3c39",
            "ecg_clean": "#2c4f7c",
            "accept_bg": "#f2f7f3",
            "abstain_bg": "#fbf3f2",
            "accept_border": "#3b7a57",
            "abstain_border": "#a64b48",
            "accept_text": "#3b7a57",
            "abstain_text": "#a64b48",
        }
    return {
        "navy": "#000000",
        "navy_border": "#000000",
        "terracotta": "#4d4d4d",
        "terracotta_border": "#000000",
        "ochre": "#8c8c8c",
        "ochre_border": "#000000",
        "slate": "#666666",
        "slate_border": "#000000",
        "green": "#000000",
        "pale_ci": "#dcdcdc",
        "abst_bar": "#c4c4c4",
        "abst_border": "#555555",
        "viol_bar": "#777777",
        "viol_border": "#000000",
        "ecg_clean": "#777777",
        "accept_bg": "#ffffff",
        "abstain_bg": "#ececec",
        "accept_border": "#000000",
        "abstain_border": "#444444",
        "accept_text": "#000000",
        "abstain_text": "#444444",
    }


def panel_title(ax, text, loc="left"):
    ax.set_title(text, loc=loc, pad=5, **BOLD)


def stress_label(label: str) -> str:
    label = re.sub(r"(\d)dB", r"\1 dB", label).replace("-3", "$-$3")
    return label.replace(" ", "\n", 1)


def draw_figure_1(out_dir: Path = DEFAULT_FIG_DIR, color: bool = True) -> None:
    """System block diagram."""
    pal = get_palette(color)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.grid(False)

    def card(x, y, w, h, title, items, fill, border="#000000", lw=1.0, ls="-", step=7.6, first=13.0, title_color="#000000"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
            facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=2,
        ))
        ax.text(x + w / 2, y + h - 4.8, title, ha="center", va="center", fontsize=9.5, color=title_color, zorder=3, **BOLD)
        ax.plot([x + 2, x + w - 2], [y + h - 9.2, y + h - 9.2], color="#cccccc", lw=0.6, zorder=3)
        for i, text in enumerate(items):
            ax.text(x + 3, y + h - first - i * step, text, ha="left", va="center", fontsize=8.5, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#000000", mutation_scale=10), zorder=4)

    top_y, top_h = 55.0, 42.0
    bot_y, bot_h = 3.0, 35.0
    card(1, top_y, 30, top_h, "1  Wearable front end",
         ["AD8232-class analog front end", "0.5 to 150 Hz, 12-bit ADC", "360 Hz, 10 s windows"], "#f8f9fa", step=8.6, first=14.5)
    card(35, top_y, 30, top_h, "2  Causal preprocessing",
         ["0.5 to 40 Hz band-pass", "or 4-level Haar SWT denoiser", "No look-ahead across windows"], "#f8f9fa", step=8.6, first=14.5)
    card(69, top_y, 30, top_h, "3  Five zero-oracle features",
         [r"$f_1$  QRS-detector disagreement", r"$f_2$  Beat-morphology dispersion",
          r"$f_3$  Baseline-wander ratio", r"$f_4$  High-frequency EMG",
          r"$f_5$  Kurtosis deficit"], "#f0f3f6", step=5.9, first=12.5)
    arrow(31, 76, 35, 76)
    arrow(65, 76, 69, 76)

    ax.plot([84, 84], [top_y, 46.5], color="#000000", lw=1.1, zorder=4)
    ax.plot([84, 16], [46.5, 46.5], color="#000000", lw=1.1, zorder=4)
    arrow(16, 46.5, 16, bot_y + bot_h)

    card(1, bot_y, 30, bot_h, "4  Composite score",
         [r"$z_j=\mathrm{clip}(f_j/s_j,\ 0,\ 6)$", r"$S_0=\frac{1}{5}\sum_j z_j$", "Fixed scales, no labels fitted"],
         "#edf1f5", step=8.6, first=14.0)
    card(35, bot_y, 30, bot_h, "5  Patient-cluster CRC",
         [r"Calibrate on $M=39$ patients", r"$\dfrac{\sum_p L_p(\hat{\tau})+1}{M+1}\leq\alpha$", "Threshold fixed before use"],
         "#edf1f5", step=8.6, first=14.0)
    arrow(31, 20.5, 35, 20.5)

    ax.plot([65, 67], [20.5, 20.5], color="#000000", lw=1.1, zorder=4)
    ax.plot([67, 67], [10.5, 30.5], color="#000000", lw=1.1, zorder=4)
    arrow(67, 30.5, 69, 30.5)
    arrow(67, 10.5, 69, 10.5)
    card(69, 23.0, 30, 15.0, "ACCEPT", [r"$S_0\leq\hat{\tau}$: expected risk $\leq\alpha$"], pal["accept_bg"], border=pal["accept_border"], lw=1.6, first=12.0, title_color=pal["accept_text"])
    card(69, 3.0, 30, 15.0, "ABSTAIN", [r"$S_0>\hat{\tau}$: flag and re-acquire"], pal["abstain_bg"], border=pal["abstain_border"], lw=1.2, ls=(0, (4, 2)), first=12.0, title_color=pal["abstain_text"])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "system_block_diagram.pdf")
    plt.close(fig)


def draw_figure_2(out_dir: Path = DEFAULT_FIG_DIR, color: bool = True) -> None:
    """Representative 10 s strips with the gate decision."""
    pal = get_palette(color)
    streams = load_mitdb_contiguous_streams(DATA_ROOT / "mitdb", n_strips_per_record=60, strip_s=10.0, start_s=60.0)
    nstdb = load_nstdb_360hz(DATA_ROOT / "nstdb", split="cal")
    corpus = generate_ambulatory_corpus(streams, nstdb, seed=2026)

    med = np.asarray(frozen["medians"], dtype=float)
    iqr = np.asarray(frozen["iqrs"], dtype=float)
    wts = np.asarray(frozen["weights"], dtype=float)
    tau = float(frozen["tau_medians"]["0.08"])

    def score(sig, fs=360.0):
        afe = apply_diagnostic_afe_360hz(sig, fs=fs)
        feat = extract_sqi_features(afe, causal_clinical_bandpass_360hz(afe, fs=fs), fs=fs)["features"]
        return float(score_uncertainty(np.asarray(feat)[np.newaxis, :], med, iqr, wts)[0])

    clean_ix = ma_ix = em_ix = None
    for i, r in enumerate(corpus):
        if clean_ix is None and r["noise_regime"] == "clean" and r["has_ectopic"] and r["group"] == "Ectopic":
            afe = apply_diagnostic_afe_360hz(r["sig_clean_mv"])
            sqi = extract_sqi_features(afe, causal_clinical_bandpass_360hz(afe))
            if not sqi["orphanidou_accept"] and sqi["morph_disp"] < 0.25 and score(r["sig_obs_mv"]) <= tau:
                clean_ix = i
        elif r["noise_regime"] in ("ma", "em") and r["snr_db"] <= 3.0 and score(r["sig_obs_mv"]) > 1.3 * tau:
            if r["noise_regime"] == "ma" and ma_ix is None:
                ma_ix = i
            if r["noise_regime"] == "em" and em_ix is None:
                em_ix = i
        if None not in (clean_ix, ma_ix, em_ix):
            break

    def snr_text(v):
        n = int(round(v))
        return f"$-${abs(n)}" if n < 0 else str(n)

    picks = [
        (f"(a) Clean ectopic strip, rec. {corpus[clean_ix]['pid']}", clean_ix, False),
        (f"(b) Muscle artifact ({snr_text(corpus[ma_ix]['snr_db'])} dB), rec. {corpus[ma_ix]['pid']}", ma_ix, True),
        (f"(c) Electrode motion ({snr_text(corpus[em_ix]['snr_db'])} dB), rec. {corpus[em_ix]['pid']}", em_ix, True),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.6), sharex=True, constrained_layout=True)
    for k, (ax, (title, ix, noisy)) in enumerate(zip(axes, picks)):
        row = corpus[ix]
        fs = float(row["fs"])
        t = np.arange(len(row["sig_obs_mv"])) / fs
        clean = causal_clinical_bandpass_360hz(apply_diagnostic_afe_360hz(row["sig_clean_mv"], fs=fs), fs=fs)
        obs = causal_clinical_bandpass_360hz(apply_diagnostic_afe_360hz(row["sig_obs_mv"], fs=fs), fs=fs)
        s0 = score(row["sig_obs_mv"], fs)

        if noisy:
            ax.plot(t, clean, color=pal["ecg_clean"], lw=1.6, alpha=0.75, label="Clean reference", zorder=2)
        ax.plot(t, obs, color="#1c1c1c", lw=0.9, label="Observed", zorder=3)

        if not noisy:
            ect_t = np.asarray(row["r_peaks_s"])[np.asarray(row["ect_flags"], dtype=bool)]
            ect_y = [obs[min(int(round(s * fs)), len(obs) - 1)] + 0.3 for s in ect_t]
            ax.plot(ect_t, ect_y, "v", color=pal["terracotta"], ms=6, label="Ectopic beat", zorder=4)

        accept = s0 <= tau
        verdict = "ACCEPT" if accept else "ABSTAIN"
        vcolor = pal["accept_text"] if accept else pal["abstain_text"]
        relation = r"\leq" if accept else ">"
        right = rf"$S_0={s0:.2f}\,{relation}\,\hat{{\tau}}={tau:.2f}$ $\rightarrow$ {verdict}"
        if not noisy:
            right += " (Orphanidou: REJECT)"
        ax.set_title(right, loc="right", pad=5, fontsize=8.0, color=vcolor)
        panel_title(ax, title)
        ax.set_ylabel("mV")
        ax.set_xlim(0, 10)
        ax.set_ylim(min(obs.min(), -1.25) - 0.15, max(obs.max(), 1.55) + 0.45)

    axes[-1].set_xlabel("Time within the 10 s window (s)")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sqi_signals_examples.pdf")
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.0f}"):
    pad = 0.015 * ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=7.5, color="#2b2b2b")


def draw_figure_3(out_dir: Path = DEFAULT_FIG_DIR, color: bool = True) -> None:
    """Subgroup preservation and stress shifts."""
    pal = get_palette(color)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.4), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    methods = [
        ("crc_10", r"Patient-cluster CRC ($\alpha=0.10$)", pal["navy"], pal["navy_border"]),
        ("orphanidou", "Orphanidou (2015)", pal["terracotta"] if color else "#ffffff", pal["terracotta_border"]),
        ("clifford", "Clifford (2012)", pal["ochre"], pal["ochre_border"]),
    ]
    bw = 0.26

    def grouped(ax, keys, group_labels, ymax):
        x = np.arange(len(keys))
        bars_all = []
        for j, (name, label, fill, edge) in enumerate(methods):
            vals = [cohort[name][k] for k in keys]
            bars = ax.bar(x + (j - 1) * bw, vals, bw, color=fill, edgecolor=edge, linewidth=0.7, alpha=0.9, label=label)
            bars_all.append((bars, "{:.0f}" if ymax > 50 else "{:.1f}"))
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, ymax)
        for bars, fmt in bars_all:
            bar_labels(ax, bars, fmt)
        ax.grid(axis="x", visible=False)

    grouped(ax_a, ["clean_sinus_accept", "clean_ectopic_accept"], ["Clean sinus", "Clean ectopic"], 110)
    ax_a.set_ylabel("Strips accepted (%)")
    panel_title(ax_a, "(a) Clean strips preserved")

    grouped(ax_b, ["risk_sinus", "risk_ectopic"], ["Sinus", "Ectopic"], 14)
    ax_b.axhline(10.0, color="#1c1c1c", ls="--", lw=1.0)
    ax_b.annotate(r"target $\alpha=10\%$", xy=(0.50, 10.0), xytext=(0.50, 12.2),
                  ha="center", va="bottom", fontsize=8, color="#444444",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666", shrinkA=2, shrinkB=2))
    ax_b.set_ylabel("Silent clinical risk (%)")
    panel_title(ax_b, "(b) Silent risk by rhythm group")

    labels = [stress_label(r["label"]) for r in stress]
    risk = np.asarray([r["crc08_risk_pct"] for r in stress])
    lo = np.asarray([r["crc08_risk_ci_lo_pct"] for r in stress])
    hi = np.asarray([r["crc08_risk_ci_hi_pct"] for r in stress])
    abst = np.asarray([r["crc08_abstain_pct"] for r in stress])
    sx = np.arange(len(labels))
    over = risk > 8.0

    ax_c.axhline(8.0, color=pal["terracotta"] if color else "#1c1c1c", ls="--", lw=1.1)
    ax_c.errorbar(sx, risk, yerr=[risk - lo, hi - risk], fmt="none", ecolor="#444444", elinewidth=1.2, capsize=3.5, zorder=3)
    ax_c.plot(sx[~over], risk[~over], "o", color=pal["navy"], mec=pal["navy_border"], mew=1.0, ms=6.5, zorder=4)
    ax_c.plot(sx[over], risk[over], "D", color=pal["terracotta"] if color else "#ffffff", mec=pal["terracotta_border"], mew=1.2, ms=7.0, zorder=4)
    ax_c.set_xticks(sx)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylim(0, max(hi) + 2)
    ax_c.set_ylabel("Silent clinical risk (%)")
    ax_c.legend(handles=[Line2D([], [], color=pal["terracotta"] if color else "#1c1c1c", ls="--", lw=1.1, label=r"target $\alpha=8\%$"),
                         Line2D([], [], marker="o", color=pal["navy"], ls="", label=r"within bound ($\leq 8\%$)"),
                         Line2D([], [], marker="D", color=pal["terracotta"] if color else "#ffffff", mec=pal["terracotta_border"], ls="", label=r"exceeds bound ($> 8\%$)")],
                loc="upper left", fontsize=8, handletextpad=0.5)
    panel_title(ax_c, r"(c) Risk under noise shift (mean, 95% CI)")

    bars = ax_d.bar(sx, abst, 0.55, color=pal["abst_bar"], edgecolor=pal["abst_border"], linewidth=0.8, alpha=0.85)
    bar_labels(ax_d, bars)
    ax_d.set_xticks(sx)
    ax_d.set_xticklabels(labels)
    ax_d.set_ylim(0, 100)
    ax_d.set_ylabel("Strips abstained (%)")
    ax_d.grid(axis="x", visible=False)
    panel_title(ax_d, "(d) Abstention under noise shift")

    handles = [Patch(facecolor=f, edgecolor=e, label=l) for _, l, f, e in methods]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cohort_and_nstdb_shift.pdf")
    plt.close(fig)


def draw_figure_4(out_dir: Path = DEFAULT_FIG_DIR, color: bool = True) -> None:
    """Risk calibration, Pareto frontier, and correlation sweep."""
    pal = get_palette(color)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.6), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    a = np.asarray(ALPHAS) * 100.0
    get = lambda block, key: np.asarray([cv[block][k][key] for k in AKEYS])
    crc_r, crc_lo, crc_hi, crc_acc = (get("crc", "patient_risk_pct"), get("crc", "patient_risk_ci_lo_pct"),
                                      get("crc", "patient_risk_ci_hi_pct"), get("crc", "accept_pct"))
    sup_r, sup_acc = get("crc_sup", "patient_risk_pct"), get("crc_sup", "accept_pct")
    iid_r, iid_acc = get("iid", "patient_risk_pct"), get("iid", "accept_pct")

    styles = {
        "crc": dict(color=pal["navy"], ls="-", marker="o", mfc=pal["navy"], mec=pal["navy_border"], label=rf"Patient-cluster CRC, $S_0$ ($M={M_CAL}$)"),
        "sup": dict(color=pal["ochre"], ls="--", marker="s", mfc=pal["ochre"], mec=pal["ochre_border"], label=rf"Patient-cluster CRC, sup. weights ($M={M_SUP}$)"),
        "iid": dict(color=pal["slate"], ls="-.", marker="^", mfc=pal["slate"], mec=pal["slate_border"], label="Window-i.i.d. calibration"),
    }

    ax_a.plot(a, a, color="#444444", ls=":", lw=1.2)
    ax_a.fill_between(a, crc_lo, crc_hi, color=pal["pale_ci"], alpha=0.55, lw=0)
    ax_a.plot(a, crc_r, **styles["crc"])
    ax_a.plot(a, sup_r, **styles["sup"])
    ax_a.plot(a, iid_r, **styles["iid"])
    ax_a.set_xlabel(r"Target risk level $\alpha$ (%)")
    ax_a.set_ylabel("Held-out patient risk (%)")
    ax_a.set_xlim(4.5, 15.5)
    ax_a.set_ylim(0, 18)
    panel_title(ax_a, "(a) Risk calibration")

    ax_b.axhline(8.0, color="#888888", ls=":", lw=1.0)
    ax_b.plot(crc_acc, crc_r, **styles["crc"])
    ax_b.plot(sup_acc, sup_r, **styles["sup"])
    ax_b.plot(iid_acc, iid_r, **styles["iid"])
    cliff_line = None
    if "heur_cliff" in cv:
        cliff_acc = get("heur_cliff", "accept_pct")
        cliff_r = get("heur_cliff", "patient_risk_pct")
        cliff_line, = ax_b.plot(cliff_acc, cliff_r, color=pal["green"], ls="--", lw=1.4, label="Clifford score with CRC")
    o, c = heur["orphanidou"], heur["clifford"]
    ax_b.plot(o["accept_pct"], o["patient_risk_pct"], "D", mfc=pal["terracotta"], mec=pal["terracotta_border"], mew=1.1, ms=6.5, zorder=6)
    ax_b.annotate("Orphanidou\n(uncalibrated)", (o["accept_pct"], o["patient_risk_pct"]),
                  xytext=(-14, 22), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=pal["terracotta_border"],
                  arrowprops=dict(arrowstyle="-|>", lw=0.7, color=pal["terracotta_border"], shrinkA=0, shrinkB=4))
    ax_b.plot(c["accept_pct"], c["patient_risk_pct"], "X", color=pal["ochre"], mec=pal["ochre_border"], mew=1.1, ms=7.0, zorder=6)
    ax_b.annotate("Clifford\n(uncalibrated)", (c["accept_pct"], c["patient_risk_pct"]),
                  xytext=(18, -22), textcoords="offset points", ha="left", va="top", fontsize=8, color=pal["ochre_border"],
                  arrowprops=dict(arrowstyle="-|>", lw=0.7, color=pal["ochre_border"], shrinkA=0, shrinkB=4))
    ax_b.text(16, 8.3, r"$\alpha=8\%$", ha="left", va="bottom", fontsize=8, color="#888888")
    ax_b.set_xlabel("Strips accepted (%)")
    ax_b.set_ylabel("Silent clinical risk (%)")
    ax_b.set_xlim(15, 90)
    ax_b.set_ylim(0, 18)
    panel_title(ax_b, "(b) Risk versus acceptance")

    corr = summary.get("correlation_sweep", [])
    sx = np.arange(len(corr))
    xl = ["i.i.d."] + [f"{r['rho']:.2f}" for r in corr[1:]]
    ax_c.axhline(8.0, color=pal["terracotta"] if color else "#1c1c1c", ls="--", lw=1.0)
    ax_c.text(0.0, 8.25, r"$\alpha=8\%$", ha="left", va="bottom", fontsize=8, color=pal["terracotta"] if color else "#1c1c1c")
    ax_c.plot(sx, [r["iid_risk_p95_pct"] for r in corr], **styles["iid"])
    ax_c.plot(sx, [r["crc_risk_p95_pct"] for r in corr], **styles["crc"])
    ax_c.set_xticks(sx)
    ax_c.set_xticklabels(xl)
    ax_c.set_xlabel(r"Markov persistence $\rho$")
    ax_c.set_ylabel(r"95th-percentile risk (%)")
    ax_c.set_ylim(0, 14)
    panel_title(ax_c, "(c) Tail risk under correlation")

    bw = 0.36
    ax_d.bar(sx - bw / 2, [r["iid_viol_pct"] for r in corr], bw, color=pal["viol_bar"], edgecolor=pal["viol_border"], linewidth=0.7, label=r"Window-i.i.d. $P(R>\alpha)$")
    ax_d.bar(sx + bw / 2, [r["crc_viol_pct"] for r in corr], bw, color=pal["navy"], edgecolor=pal["navy_border"], linewidth=0.7, label=r"Patient-CRC $P(R>\alpha)$")
    ax_d.set_xticks(sx)
    ax_d.set_xticklabels(xl)
    ax_d.set_xlabel(r"Markov persistence $\rho$")
    ax_d.set_ylabel(r"Calibrations with risk $>\alpha$ (%)")
    ax_d.set_ylim(0, 75)
    ax_d.grid(axis="x", visible=False)
    panel_title(ax_d, r"(d) Exceedance probability ($\alpha=8\%$)")

    handles = [Line2D([], [], color="#444444", ls=":", lw=1.2, label=r"Target risk bound $\alpha$")]
    handles += [Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], label=st["label"])
               for st in styles.values()]
    if cliff_line is not None:
        handles.append(Line2D([], [], color=pal["green"], ls="--", lw=1.4, label="Clifford score with CRC"))
    handles.append(Patch(facecolor=pal["pale_ci"], alpha=0.6, label="95% CI of CRC"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "crc_validity_pareto.pdf")
    plt.close(fig)


def render_all(out_dir: Path, color: bool = True) -> None:
    draw_figure_1(out_dir, color=color)
    draw_figure_2(out_dir, color=color)
    draw_figure_3(out_dir, color=color)
    draw_figure_4(out_dir, color=color)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grayscale", action="store_true", help="Render in grayscale instead of default color")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR, help="Output directory (default: results/figures)")
    args = ap.parse_args()

    color = not args.grayscale
    mode_str = "grayscale" if args.grayscale else "color"
    print(f"Rendering {mode_str} figures to {args.out_dir}...")
    render_all(args.out_dir, color=color)


if __name__ == "__main__":
    main()
