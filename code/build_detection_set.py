"""build_detection_set.py - one row per skill version with text and before/after labels, for the detectors.

Labels per skill version (key = slug@version):
  ss_old / ss_new   SkillSpector flag in the last snapshot <= 2026-09-07 / first snapshot >= 2026-09-21
  vt_old / vt_new   VirusTotal flag (suspicious|malicious), same two dates  (control: barely drifts)
  st_old / st_new   static-analysis flag, same two dates                      (control)
  verdict_any       registry verdict suspicious|malicious in any snapshot     (rare, secondary task)
Texts are joined on Colab (notebook 2) by md_sha / bundle_sha; this file holds labels only (small).
Split (by skill slug, so versions of one skill never cross): time-based on first appearance
  train: first seen <= 2026-07-31   valid: 2026-08-01..08-15   test: 2026-08-16..09-07
Only versions observed both before and after the September rescan get old/new labels.
"""
import os

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "data", "labels.parquet")

con = duckdb.connect()
con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{os.path.join(ROOT, 'data', '.duckdb_tmp')}'")
con.execute(f"""
CREATE TABLE lab AS
WITH p AS (
  SELECT skill_slug, skill_slug||'@'||skill_version k, snapshot, md_sha, bundle_sha,
    (skillspector_status='suspicious') ss,
    (virustotal_status IN ('suspicious','malicious')) vt,
    (static_status IN ('suspicious','malicious')) st,
    (clawscan_verdict IN ('suspicious','malicious')) vd
  FROM read_parquet('{D}/panel.parquet'))
SELECT k, any_value(skill_slug) slug, min(snapshot) first_seen, max(snapshot) last_seen,
  arg_max(md_sha, snapshot) md_sha, arg_max(bundle_sha, snapshot) bundle_sha,
  arg_max(ss, snapshot) FILTER (WHERE snapshot<='2026-09-07') ss_old,
  arg_min(ss, snapshot) FILTER (WHERE snapshot>='2026-09-21') ss_new,
  arg_max(vt, snapshot) FILTER (WHERE snapshot<='2026-09-07') vt_old,
  arg_min(vt, snapshot) FILTER (WHERE snapshot>='2026-09-21') vt_new,
  arg_max(st, snapshot) FILTER (WHERE snapshot<='2026-09-07') st_old,
  arg_min(st, snapshot) FILTER (WHERE snapshot>='2026-09-21') st_new,
  bool_or(vd) verdict_any
FROM p GROUP BY k
""")
con.execute("""
CREATE TABLE lab2 AS
SELECT *, CASE
  WHEN first_seen<='2026-07-31' THEN 'train'
  WHEN first_seen<='2026-08-15' THEN 'valid'
  WHEN first_seen<='2026-09-07' THEN 'test'
  ELSE 'late' END AS split
FROM lab
""")
# a slug whose versions span several splits goes entirely to its earliest split (no leakage across splits)
con.execute("""
CREATE TABLE lab3 AS
SELECT l.* EXCLUDE (split), s.split
FROM lab2 l JOIN (
  SELECT slug, CASE min(CASE split WHEN 'train' THEN 0 WHEN 'valid' THEN 1 WHEN 'test' THEN 2 ELSE 3 END)
    WHEN 0 THEN 'train' WHEN 1 THEN 'valid' WHEN 2 THEN 'test' ELSE 'late' END split
  FROM lab2 GROUP BY slug) s USING (slug)
""")
con.execute(f"COPY lab3 TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)")
print(con.sql(f"""
SELECT split, count(*) n, count(ss_old) with_old, count(ss_new) with_new,
  sum(ss_old::INT) ss_old_pos, sum(ss_new::INT) ss_new_pos,
  sum((ss_old IS NOT NULL AND ss_new IS NOT NULL AND ss_old<>ss_new)::INT) ss_changed,
  sum((vt_old IS NOT NULL AND vt_new IS NOT NULL AND vt_old<>vt_new)::INT) vt_changed,
  sum((st_old IS NOT NULL AND st_new IS NOT NULL AND st_old<>st_new)::INT) st_changed,
  sum(vt_old::INT) vt_pos, sum(st_old::INT) st_pos, sum(verdict_any::INT) verdict_pos,
  count(DISTINCT md_sha) uniq_md
FROM '{OUT}' GROUP BY 1 ORDER BY 1"""))
print(round(os.path.getsize(OUT) / 1e6, 1), "MB ->", OUT)
