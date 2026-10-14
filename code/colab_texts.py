"""colab_texts.py - fetch the skill texts needed by labels.parquet straight from Hugging Face (run on Colab).

Walks the daily snapshots newest -> oldest, hashes each row exactly like build_snapshots.py, and keeps the
(truncated) SKILL.md and bundle text of every md_sha / bundle_sha listed in labels.parquet. Stops early once
everything is found. Returns labels joined with md_text / bundle_text.
"""
import json
import os
import time

import pandas as pd

import build_snapshots as bs


def fetch_texts(labels: pd.DataFrame, work_dir: str, md_max=12_000, bundle_max=8_000, days=None,
                download=None):
    need_md, need_b = set(labels.md_sha.dropna()), set(labels.bundle_sha.dropna())
    md_txt, b_txt = {}, {}
    if days is None:
        days = bs.daily_commits()
    if download is None:
        from huggingface_hub import hf_hub_download

        def download(commit):
            return hf_hub_download(bs.REPO, "data/latest.jsonl", repo_type="dataset", revision=commit,
                                   cache_dir=work_dir)
    for day, commit, _ in sorted(days, reverse=True):
        if not (need_md - md_txt.keys()) and not (need_b - b_txt.keys()):
            break
        t0 = time.time()
        try:
            path = download(commit)
        except Exception as e:
            bs.log(f"{day}: no data ({type(e).__name__})")
            continue
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                md = obj.get("skill_md_content") or ""
                h = bs.sha(md)
                if h in need_md and h not in md_txt:
                    md_txt[h] = md[:md_max]
                btext, bsig, _, _ = bs.bundle_text(bs._list(obj.get("skill_bundle_content")))
                if bsig in need_b and bsig not in b_txt:
                    b_txt[bsig] = btext[:bundle_max]
        if os.path.getsize(path) > 50_000_000 and os.path.realpath(path) != path:
            os.remove(os.path.realpath(path))  # free disk (HF cache blob)
        bs.log(f"{day}: md {len(md_txt)}/{len(need_md)}, bundle {len(b_txt)}/{len(need_b)} ({time.time() - t0:.0f}s)")
    out = labels.copy()
    out["md_text"] = out.md_sha.map(md_txt).fillna("")
    out["bundle_text"] = out.bundle_sha.map(b_txt).fillna("")
    out["text_found"] = out.md_sha.isin(md_txt.keys())
    bs.log(f"texts found for {out.text_found.mean():.1%} of skill versions")
    return out
