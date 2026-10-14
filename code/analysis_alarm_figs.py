"""analysis_alarm_figs.py - canary label-drift alarm + publication figures and tables.

Canary alarm: a fixed random set of skill versions present in every snapshot is re-checked at each snapshot;
the share whose scanner label changed since the previous snapshot is the "flip rate". An alarm fires when the
flip rate exceeds mean + 3 SD of the earlier steps (needs >= 5 earlier steps). Power and false alarms are
estimated over 1,000 random canary sets per size.

Outputs: results/final/{fig*.pdf,png, table*.csv/.tex, numbers.json}
Light on memory: duckdb with a 1.2 GB limit; only small matrices in pandas.
"""
import glob
import json
import os

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PANEL = os.path.join(ROOT, "data", "panel.parquet")
DET = os.path.join(ROOT, "results", "detectors")
OUT = os.path.join(ROOT, "results", "final")
os.makedirs(OUT, exist_ok=True)
NUM = {}

con = duckdb.connect()
con.execute("SET memory_limit='1200MB'")
con.execute("SET threads=2")
con.execute(f"""CREATE VIEW p AS SELECT skill_slug||'@'||skill_version k, snapshot,
  coalesce(skillspector_status='suspicious', false)::INT ss,
  coalesce(static_status IN ('suspicious','malicious'), false)::INT st,
  coalesce(virustotal_status IN ('suspicious','malicious'), false)::INT vt,
  coalesce(clawscan_verdict IN ('suspicious','malicious'), false)::INT vd
  FROM read_parquet('{PANEL}')""")

snaps = [r[0] for r in con.sql("SELECT DISTINCT snapshot FROM p ORDER BY 1").fetchall()]
NUM["snapshots"] = len(snaps)

# ---------------------------------------------------------------- per-snapshot overview
ov = con.sql("""SELECT snapshot, count(*) n, avg(ss)*100 ss_pct, avg(st)*100 st_pct, avg(vt)*100 vt_pct,
  avg(vd)*100 vd_pct FROM p GROUP BY 1 ORDER BY 1""").df()
ov.to_csv(os.path.join(OUT, "table_overview_by_snapshot.csv"), index=False)
agr = pd.read_csv(os.path.join(ROOT, "results", "drift", "rq3_agreement.csv"))

# ---------------------------------------------------------------- canary matrix
full = con.sql(f"""SELECT k FROM p GROUP BY k HAVING count(DISTINCT snapshot)={len(snaps)}""").df().k
NUM["versions_in_all_snapshots"] = int(len(full))
mats = {}
con.register("fullk", pd.DataFrame({"k": full}))
for lab in ("ss", "st", "vt", "vd"):
    m = con.sql(f"SELECT k, snapshot, {lab} v FROM p WHERE k IN (SELECT k FROM fullk)").df()
    mats[lab] = m.pivot(index="k", columns="snapshot", values="v")[snaps].to_numpy(np.int8)
flip = {lab: (np.diff(M, axis=1) != 0) for lab, M in mats.items()}  # versions x steps
steps = snaps[1:]
flip_rate = pd.DataFrame({lab: f.mean(0) * 100 for lab, f in flip.items()}, index=steps)
flip_rate.index.name = "snapshot"
flip_rate.to_csv(os.path.join(OUT, "table_canary_flip_rate_full.csv"))


def alarms(rate, k=3.0, warm=5):
    """Indices of steps where rate > mean+k*sd of all earlier steps (after a warm-up)."""
    out = []
    for i in range(warm, len(rate)):
        hist = rate[:i]
        if rate[i] > hist.mean() + k * max(hist.std(), 1e-9):
            out.append(i)
    return out


# Alarm rule: fire when the share of canary versions whose label flipped since the previous snapshot >= tau.
# Truth per step = full-population flip rate (all versions present in every snapshot):
#   real event if >= tau, quiet if < tau / 10.  Power = alarms on real events; false alarm = any alarm on a quiet step.
power = []
rng = np.random.default_rng(0)
for tau in (0.5, 5.0):
    for lab in ("ss", "vd"):
        full_rate = flip[lab].mean(0) * 100
        real = np.flatnonzero(full_rate >= tau)
        quiet = np.flatnonzero(full_rate < tau / 10)
        for size in (50, 100, 200, 500, 1000):
            hits, fa = np.zeros(len(real)), 0
            for _ in range(1000):
                idx = rng.choice(len(full), size, replace=False)
                rate = flip[lab][idx].mean(0) * 100
                fire = rate >= tau
                hits += fire[real]
                fa += bool(fire[quiet].any())
            power.append({"tau_pct": tau, "label": lab, "canary_size": size,
                          "real_events": ", ".join(steps[i] for i in real) or "none",
                          "detection_rate_per_event": ", ".join(f"{h / 1000:.3f}" for h in hits) or "-",
                          "any_false_alarm_rate": fa / 1000})
power = pd.DataFrame(power)
power.to_csv(os.path.join(OUT, "table_canary_power.csv"), index=False)
NUM["canary_power"] = power.to_dict("records")
NUM["full_canary_flip_rate_pct"] = {lab: {d: round(float(v), 3) for d, v in flip_rate[lab].items() if v > 0}
                                    for lab in flip_rate.columns}

# ---------------------------------------------------------------- detector results
det = {}
for f in glob.glob(os.path.join(DET, "*.json")):
    det[os.path.basename(f)[:-5]] = json.load(open(f))
rows = []
for name, r in sorted(det.items()):
    if "trained_on_old" not in r:
        continue
    model, src, _ = name.split("_")
    for tr in ("old", "new"):
        for ev in ("old", "new"):
            m = r[f"trained_on_{tr}"][f"test_vs_{ev}"]
            rows.append({"model": model, "labels": src, "trained_on": tr, "evaluated_on": ev, "n": m["n"],
                         "positives": m["pos"], "roc_auc": m.get("roc_auc"), "auc_lo": m["roc_auc_ci"][0],
                         "auc_hi": m["roc_auc_ci"][1], "pr_auc": m.get("pr_auc"), "f1": m["f1"],
                         "precision": m["precision"], "recall": m["recall"]})
dres = pd.DataFrame(rows)
dres.to_csv(os.path.join(OUT, "table_detectors.csv"), index=False)
for name in ("tfidf_ss_s0", "bert_ss_s0"):
    if name in det:
        NUM[f"{name}_test_labels_flipped"] = det[name]["test_label_changed"]
        NUM[f"{name}_test_n"] = det[name]["sizes"]["test"]
for name in ("tfidf_verdict_s0", "bert_verdict_s0"):
    if name in det:
        NUM[name] = {k: det[name]["test"].get(k) for k in ("pos", "n", "roc_auc", "pr_auc", "f1")}

# ranking check: does the model ranking change between old and new labels?
rk = dres[(dres.labels == "ss") & (dres.trained_on == "old")].pivot(index="model", columns="evaluated_on",
                                                                     values="roc_auc")
NUM["ss_auc_by_model_trained_old"] = rk.round(4).to_dict("index")

# ---------------------------------------------------------------- figures
PAL = {"ss": "#2a78d6", "st": "#eb6834", "vt": "#1baf7a", "vd": "#4a3aa7"}
NAMES = {"ss": "SkillSpector", "st": "Static analysis", "vt": "VirusTotal", "vd": "Registry verdict"}
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
dates = pd.to_datetime(ov.snapshot)
ev_date = pd.Timestamp("2026-09-14")


def style(ax, ylabel):
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.axvspan(pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-21"), color="#f0efec", zorder=0)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


# Fig 1: flag rates over time + registry size
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for lab, col in (("ss", "ss_pct"), ("st", "st_pct"), ("vt", "vt_pct"), ("vd", "vd_pct")):
    a1.plot(dates, ov[col], color=PAL[lab], lw=1.6, marker="o", ms=2.5, label=NAMES[lab])
style(a1, "Flagged skills (%)")
a1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
a2.plot(dates, ov.n / 1000, color="#52514e", lw=1.6, marker="o", ms=2.5)
style(a2, "Skills (k)")
a2.tick_params(axis="x", rotation=0)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_flag_rates.{e}"), dpi=300)
plt.close(fig)

# Fig 2: scanner agreement
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ad = pd.to_datetime(agr.snapshot)
for col, lab, c in (("jaccard_static_ss", "Jaccard static–SkillSpector", PAL["st"]),
                    ("jaccard_vt_ss", "Jaccard VirusTotal–SkillSpector", PAL["vt"]),
                    ("jaccard_static_vt", "Jaccard static–VirusTotal", PAL["vd"]),
                    ("all_three_share", "Flagged by all three", PAL["ss"])):
    ax.plot(ad, agr[col] * 100, lw=1.6, marker="o", ms=2.5, color=c, label=lab)
style(ax, "Agreement (%)")
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig2_agreement.{e}"), dpi=300)
plt.close(fig)

# Fig 3: canary flip rate (full canary) on log scale + alarm line for SkillSpector
fig, ax = plt.subplots(figsize=(3.5, 2.3))
sd = pd.to_datetime(steps)
for lab in ("ss", "st", "vt", "vd"):
    ax.plot(sd, flip_rate[lab] + 1e-3, lw=1.6, marker="o", ms=2.5, color=PAL[lab], label=NAMES[lab])
ax.set_yscale("log")
style(ax, "Labels flipped (%, log scale)")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig3_canary_flips.{e}"), dpi=300)
plt.close(fig)

# Fig 4: same frozen detector, scored against pre- vs post-rescan labels
sub = dres[(dres.trained_on == "old") & (dres.labels.isin(["ss", "st"]))]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
mets = ["roc_auc", "f1", "precision", "recall"]
for ax, model in zip(axes, ("tfidf", "bert")):
    x = np.arange(len(mets))
    w = 0.2
    for j, (src, ev, c, h, lab) in enumerate((("ss", "old", PAL["ss"], "", "SkillSpector, old labels"),
                                              ("ss", "new", PAL["st"], "///", "SkillSpector, new labels"),
                                              ("st", "old", PAL["vt"], "...", "Static (control), old"),
                                              ("st", "new", PAL["vd"], "xx", "Static (control), new"))):
        r = sub[(sub.model == model) & (sub.labels == src) & (sub.evaluated_on == ev)]
        if r.empty:
            continue
        vals = [float(r[m].iloc[0]) for m in mets]
        ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=c, hatch=h, edgecolor="white", lw=0.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["ROC-AUC", "F1", "Precision", "Recall"])
    ax.set_title({"tfidf": "TF-IDF + logistic regression", "bert": "ModernBERT (fine-tuned)"}[model],
                 loc="left", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Score")
axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08))
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig4_detector_label_drift.{e}"), dpi=300)
plt.close(fig)

json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=str)
print(json.dumps(NUM, indent=1, default=str))
print(power.to_string(index=False))
