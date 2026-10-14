"""build_snapshots.py - rebuild every historical snapshot of OpenClaw/clawhub-security-signals-live.

For each day the dataset was published (one Hugging Face commit per day), stream data/latest.jsonl
at that exact commit and keep:
  panel.parquet   one row per (snapshot, skill version): verdicts and scanner signals, content hashes
  texts.parquet   deduplicated texts keyed by sha256 (SKILL.md, bundle, ClawScan summary, ClawScan context)
  snapshots.json  per snapshot: commit, date, manifest, row counts, parse problems

Raw files are deleted after parsing, so disk use stays at one snapshot (~3 GB) at a time.
Everything is pinned to commit SHAs, so the build is reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

REPO = "OpenClaw/clawhub-security-signals-live"
BUNDLE_MAX_CHARS = 60_000  # per skill version, for the text model; full size is recorded separately

PANEL_COLS = [
    "snapshot", "commit", "id", "skill_slug", "skill_version",
    "md_sha", "md_chars", "bundle_sha", "bundle_files", "bundle_bytes",
    "clawscan_verdict", "clawscan_confidence", "clawscan_model", "summary_sha", "context_sha",
    "static_status", "static_finding_count", "static_reason_codes",
    "virustotal_status", "virustotal_malicious_count", "virustotal_suspicious_count",
    "virustotal_harmless_count", "virustotal_undetected_count",
    "skillspector_status", "skillspector_score", "skillspector_severity", "skillspector_issue_count",
    "skillspector_issue_codes", "skillspector_issue_categories",
]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def _list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return [v]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _int(v):
    try:
        return None if v is None or v == "" else int(v)
    except (TypeError, ValueError):
        return None


def bundle_text(files):
    """Concatenate bundled files as '### path' blocks (sorted by path), plus a content signature."""
    files = sorted((f for f in files if isinstance(f, dict)), key=lambda f: str(f.get("path", "")))
    parts, sig, total = [], [], 0
    for f in files:
        path, content = str(f.get("path", "")), f.get("content")
        content = "" if content is None else str(content)
        sig.append(f"{path}:{f.get('sha256') or sha(content)}")
        total += _int(f.get("sizeBytes")) or len(content)
        parts.append(f"### {path}\n{content}")
    return "\n\n".join(parts), sha("\n".join(sig)), len(files), total


def parse_row(obj, snapshot, commit):
    """One JSONL object -> (panel row, {sha: (kind, text)})."""
    md = obj.get("skill_md_content") or ""
    btext, bsig, nfiles, nbytes = bundle_text(_list(obj.get("skill_bundle_content")))
    summary = obj.get("clawscan_summary") or ""
    ctx = obj.get("clawscan_context")
    ctx = ctx if isinstance(ctx, str) else json.dumps(ctx, sort_keys=True, ensure_ascii=False)
    md_sha, sum_sha, ctx_sha = sha(md), sha(summary), sha(ctx or "")
    row = {
        "snapshot": snapshot, "commit": commit,
        "id": obj.get("id"), "skill_slug": obj.get("skill_slug"), "skill_version": obj.get("skill_version"),
        "md_sha": md_sha, "md_chars": len(md), "bundle_sha": bsig, "bundle_files": nfiles, "bundle_bytes": nbytes,
        "clawscan_verdict": obj.get("clawscan_verdict"), "clawscan_confidence": obj.get("clawscan_confidence"),
        "clawscan_model": obj.get("clawscan_model"), "summary_sha": sum_sha, "context_sha": ctx_sha,
        "static_status": obj.get("static_status"), "static_finding_count": _int(obj.get("static_finding_count")),
        "static_reason_codes": "|".join(map(str, _list(obj.get("static_reason_codes")))),
        "virustotal_status": obj.get("virustotal_status"),
        "virustotal_malicious_count": _int(obj.get("virustotal_malicious_count")),
        "virustotal_suspicious_count": _int(obj.get("virustotal_suspicious_count")),
        "virustotal_harmless_count": _int(obj.get("virustotal_harmless_count")),
        "virustotal_undetected_count": _int(obj.get("virustotal_undetected_count")),
        "skillspector_status": obj.get("skillspector_status"),
        "skillspector_score": _int(obj.get("skillspector_score")),
        "skillspector_severity": obj.get("skillspector_severity"),
        "skillspector_issue_count": _int(obj.get("skillspector_issue_count")),
        "skillspector_issue_codes": "|".join(map(str, _list(obj.get("skillspector_issue_codes")))),
        "skillspector_issue_categories": "|".join(map(str, _list(obj.get("skillspector_issue_categories")))),
    }
    texts = {md_sha: ("md", md), bsig: ("bundle", btext[:BUNDLE_MAX_CHARS]),
             sum_sha: ("summary", summary), ctx_sha: ("context", ctx or "")}
    return row, texts


def parse_jsonl(path, snapshot, commit, seen_texts, text_sink):
    """Stream a JSONL file. New texts go to text_sink (list); returns (rows, stats)."""
    rows, bad, n = [], 0, 0
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            row, texts = parse_row(obj, snapshot, commit)
            rows.append(row)
            for h, (kind, t) in texts.items():
                if h not in seen_texts:
                    seen_texts.add(h)
                    text_sink.append({"sha": h, "kind": kind, "text": t})
    return rows, {"lines": n, "bad_lines": bad, "rows": len(rows)}


def daily_commits():
    """Last commit of each day, oldest first: [(YYYY-MM-DD, sha, title)]."""
    from huggingface_hub import HfApi
    commits = HfApi().list_repo_commits(REPO, repo_type="dataset")
    by_day = {}
    for c in sorted(commits, key=lambda c: c.created_at):
        by_day[c.created_at.strftime("%Y-%m-%d")] = (c.commit_id, c.title)
    return [(d, s, t) for d, (s, t) in sorted(by_day.items())]


def build(out_dir, work_dir, limit=None):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    days = daily_commits()
    if limit:
        days = days[:limit]
    log(f"{len(days)} daily snapshots: {days[0][0]} -> {days[-1][0]}")
    seen, all_rows, text_parts, meta = set(), [], [], []
    for i, (day, commit, title) in enumerate(days, 1):
        t0 = time.time()
        try:
            data = hf_hub_download(REPO, "data/latest.jsonl", repo_type="dataset", revision=commit,
                                   cache_dir=work_dir)
        except Exception as e:  # a commit without the data file (e.g. the initial commit)
            log(f"[{i}/{len(days)}] {day} {commit[:8]}: no data file ({type(e).__name__}), skipped")
            meta.append({"snapshot": day, "commit": commit, "title": title, "skipped": str(e)[:200]})
            continue
        try:
            mpath = hf_hub_download(REPO, "metadata/latest-manifest.json", repo_type="dataset", revision=commit,
                                    cache_dir=work_dir)
            manifest = json.load(open(mpath))
        except Exception as e:
            manifest = {"error": str(e)[:200]}
        sink = []
        rows, stats = parse_jsonl(data, day, commit, seen, sink)
        all_rows.extend(rows)
        if sink:
            part = os.path.join(work_dir, f"texts_{i:03d}.parquet")
            pd.DataFrame(sink).to_parquet(part, compression="zstd", index=False)
            text_parts.append(part)
        meta.append({"snapshot": day, "commit": commit, "title": title, "manifest": manifest,
                     "file_bytes": os.path.getsize(data), "new_texts": len(sink), **stats})
        # free disk: remove downloaded blobs for this revision
        for root, _, files in os.walk(work_dir):
            for fn in files:
                p = os.path.join(root, fn)
                if not fn.startswith("texts_") and os.path.getsize(p) > 50_000_000:
                    os.remove(p)
        log(f"[{i}/{len(days)}] {day} {commit[:8]}: {stats['rows']} rows, {stats['bad_lines']} bad, "
            f"{len(sink)} new texts, {time.time() - t0:.0f}s")
    panel = pd.DataFrame(all_rows, columns=PANEL_COLS)
    panel.to_parquet(os.path.join(out_dir, "panel.parquet"), compression="zstd", index=False)
    texts = pd.concat([pd.read_parquet(p) for p in text_parts], ignore_index=True) if text_parts else pd.DataFrame()
    texts.to_parquet(os.path.join(out_dir, "texts.parquet"), compression="zstd", index=False)
    json.dump(meta, open(os.path.join(out_dir, "snapshots.json"), "w"), indent=1, default=str)
    log(f"done: panel {panel.shape}, texts {texts.shape}")
    return panel, texts, meta


if __name__ == "__main__":
    # local test on an already-downloaded JSONL:  python build_snapshots.py --test file.jsonl
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        seen, sink = set(), []
        rows, stats = parse_jsonl(sys.argv[2], "test", "local", seen, sink)
        print(stats, "texts:", len(sink))
        import collections
        print(collections.Counter(r["clawscan_verdict"] for r in rows))
        print(rows[0])
    else:
        build(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
