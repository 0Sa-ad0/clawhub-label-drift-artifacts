# Label drift in an AI-agent skill registry: data and code

Artifacts for the paper *"When the Labels Move: Silent Scanner Drift in an AI-Agent Skill Registry and Its Effect on Detector Evaluation"*.

Source data: [OpenClaw/clawhub-security-signals-live](https://huggingface.co/datasets/OpenClaw/clawhub-security-signals-live) (MIT license), 21 snapshots from 2026-06-30 to 2026-10-12, each pinned by its Hugging Face commit.

## Contents

| Path | What it is |
|---|---|
| `code/build_snapshots.py` | Rebuilds all 21 snapshots from the pinned commits into `data/panel.parquet` (labels) and `data/texts.parquet` (skill texts) |
| `code/analysis_rq123.py` | Label changes, rescan analysis, scanner agreement per snapshot → `results/drift/` |
| `code/build_detection_set.py` | Per skill version: old (≤ 2026-09-07) and new (≥ 2026-09-21) labels, time-based splits → `data/labels.parquet` |
| `code/colab_texts.py`, `code/detect.py` | Fetch skill texts by content hash; TF-IDF and ModernBERT detectors trained on old/new labels → `results/detectors/` |
| `code/analysis_alarm_figs.py` | Canary alarm evaluation, tables and figures → `results/final/` |
| `notebooks/` | Google Colab notebooks used to run the snapshot build (CPU) and the detectors (GPU) |
| `data/labels.parquet` | Label table used by the detectors (included) |
| `results/` | All result files reported in the paper |

## Reproduce

1. Build the snapshots (about 1 hour on a Colab CPU runtime): run `notebooks/notebook1_build_snapshots.ipynb`, or `python code/build_snapshots.py data work`. Put `panel.parquet` and `texts.parquet` in `data/`.
2. `python code/analysis_rq123.py`
3. `python code/build_detection_set.py`
4. Detectors (GPU recommended): `notebooks/notebook2_detectors.ipynb`
5. `python code/analysis_alarm_figs.py`

Requirements: Python 3.12, `duckdb`, `pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `huggingface_hub`, and for the transformer detector `torch` and `transformers`.

Re-running steps 2 and 5 on the rebuilt panel reproduces the files in `results/` exactly.

## License

Code: MIT (see `LICENSE`). Derived data follow the MIT license of the source dataset.
