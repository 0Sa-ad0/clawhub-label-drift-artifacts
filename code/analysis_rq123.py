"""analysis_rq123.py - longitudinal analysis of ClawHub security signals (21 snapshots, 30 Jun - 12 Oct 2026).

RQ1 label instability : does the verdict of an UNCHANGED skill version (same SKILL.md and bundle hashes) change?
RQ2 scanner lead-lag  : for skill versions first seen unflagged and later flagged, which signal fires first?
RQ3 agreement drift   : pairwise Jaccard / all-three / single-scanner share of the three scanners per snapshot.

Input : data/panel.parquet (built by build_snapshots.py; row counts match every manifest).
Output: results/drift/{*.csv, summary.json}
"""
import json
import os

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PANEL = os.path.join(ROOT, "data", "panel.parquet")
OUT = os.path.join(ROOT, "results", "drift")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=4")

# Flag definitions (positive = needs review).
con.execute(f"""
CREATE VIEW p AS
SELECT *,
  skill_slug || '@' || skill_version AS k,
  (clawscan_verdict IN ('suspicious','malicious'))           AS f_verdict,
  (static_status IN ('suspicious','malicious'))              AS f_static,
  (virustotal_status IN ('suspicious','malicious'))          AS f_vt,
  (skillspector_status = 'suspicious')                       AS f_ss
FROM read_parquet('{PANEL}')
""")
summary = {}


def save(name, sql):
    df = con.sql(sql).df()
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    return df


# ---------------- RQ1: label instability of unchanged skill versions ----------------
con.execute("""
CREATE TABLE seq AS
SELECT k, snapshot, md_sha, bundle_sha, clawscan_verdict, clawscan_model, clawscan_confidence,
       static_status, virustotal_status, skillspector_status,
       LAG(snapshot)            OVER w AS prev_snapshot,
       LAG(md_sha)              OVER w AS prev_md, LAG(bundle_sha) OVER w AS prev_bundle,
       LAG(clawscan_verdict)    OVER w AS prev_verdict,
       LAG(clawscan_model)      OVER w AS prev_model,
       LAG(clawscan_confidence) OVER w AS prev_conf,
       LAG(static_status)       OVER w AS prev_static,
       LAG(virustotal_status)   OVER w AS prev_vt,
       LAG(skillspector_status) OVER w AS prev_ss
FROM p WINDOW w AS (PARTITION BY k ORDER BY snapshot)
""")
con.execute("CREATE VIEW steps AS SELECT * FROM seq WHERE prev_snapshot IS NOT NULL")

r = con.sql("""
SELECT count(*) steps,
  sum(CASE WHEN md_sha<>prev_md OR bundle_sha<>prev_bundle THEN 1 ELSE 0 END) content_changed,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_verdict IS DISTINCT FROM prev_verdict THEN 1 ELSE 0 END) verdict_changed_same_content,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_model IS DISTINCT FROM prev_model THEN 1 ELSE 0 END) model_changed_same_content,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND static_status IS DISTINCT FROM prev_static THEN 1 ELSE 0 END) static_changed_same_content,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND virustotal_status IS DISTINCT FROM prev_vt THEN 1 ELSE 0 END) vt_changed_same_content,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND skillspector_status IS DISTINCT FROM prev_ss THEN 1 ELSE 0 END) ss_changed_same_content
FROM steps""").df().iloc[0].to_dict()
summary["rq1_steps"] = {k: int(v) for k, v in r.items()}

# per skill version: ever changed verdict while content stayed identical over its whole life
r = con.sql("""
WITH life AS (
  SELECT k, count(*) n_snap, count(DISTINCT md_sha) n_md, count(DISTINCT bundle_sha) n_b,
         count(DISTINCT coalesce(clawscan_verdict,'<null>')) n_verdict,
         count(DISTINCT coalesce(static_status,'<null>')) n_static,
         count(DISTINCT coalesce(virustotal_status,'<null>')) n_vt,
         count(DISTINCT coalesce(skillspector_status,'<null>')) n_ss,
         bool_or(f_verdict) ever_flagged
  FROM p GROUP BY k)
SELECT count(*) versions_multi,
  sum(CASE WHEN n_md=1 AND n_b=1 THEN 1 ELSE 0 END) unchanged_content,
  sum(CASE WHEN n_md=1 AND n_b=1 AND n_verdict>1 THEN 1 ELSE 0 END) unchanged_verdict_changed,
  sum(CASE WHEN n_md=1 AND n_b=1 AND n_static>1 THEN 1 ELSE 0 END) unchanged_static_changed,
  sum(CASE WHEN n_md=1 AND n_b=1 AND n_vt>1 THEN 1 ELSE 0 END) unchanged_vt_changed,
  sum(CASE WHEN n_md=1 AND n_b=1 AND n_ss>1 THEN 1 ELSE 0 END) unchanged_ss_changed,
  sum(CASE WHEN n_md=1 AND n_b=1 AND ever_flagged THEN 1 ELSE 0 END) unchanged_ever_flagged,
  sum(CASE WHEN n_md=1 AND n_b=1 AND ever_flagged AND n_verdict>1 THEN 1 ELSE 0 END) unchanged_flagged_and_changed
FROM life WHERE n_snap>=2""").df().iloc[0].to_dict()
summary["rq1_versions"] = {k: int(v) for k, v in r.items()}

save("rq1_verdict_transitions", """
SELECT coalesce(prev_verdict,'<null>') AS from_verdict, coalesce(clawscan_verdict,'<null>') AS to_verdict,
       count(*) n,
       sum(CASE WHEN clawscan_model IS DISTINCT FROM prev_model THEN 1 ELSE 0 END) with_model_change
FROM steps WHERE md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_verdict IS DISTINCT FROM prev_verdict
GROUP BY 1,2 ORDER BY n DESC""")
save("rq1_changes_by_snapshot", """
SELECT snapshot, count(*) steps,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_verdict IS DISTINCT FROM prev_verdict THEN 1 ELSE 0 END) verdict_flips,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_model IS DISTINCT FROM prev_model THEN 1 ELSE 0 END) model_changes,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND virustotal_status IS DISTINCT FROM prev_vt THEN 1 ELSE 0 END) vt_changes,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND skillspector_status IS DISTINCT FROM prev_ss THEN 1 ELSE 0 END) ss_changes,
  sum(CASE WHEN md_sha=prev_md AND bundle_sha=prev_bundle AND static_status IS DISTINCT FROM prev_static THEN 1 ELSE 0 END) static_changes
FROM steps GROUP BY 1 ORDER BY 1""")
save("rq1_model_transitions", """
SELECT coalesce(prev_model,'<null>') from_model, coalesce(clawscan_model,'<null>') to_model, count(*) n,
       sum(CASE WHEN clawscan_verdict IS DISTINCT FROM prev_verdict THEN 1 ELSE 0 END) verdict_also_changed
FROM steps WHERE md_sha=prev_md AND bundle_sha=prev_bundle AND clawscan_model IS DISTINCT FROM prev_model
GROUP BY 1,2 ORDER BY n DESC""")

# ---------------- RQ2: lead-lag of signals on versions first seen unflagged ----------------
save("rq2_first_flag", """
WITH first AS (
  SELECT k, min(snapshot) s0 FROM p GROUP BY k),
base AS (
  SELECT p.* FROM p JOIN first f ON p.k=f.k AND p.snapshot=f.s0
  WHERE NOT coalesce(f_verdict,false) AND NOT coalesce(f_static,false) AND NOT coalesce(f_vt,false)
        AND NOT coalesce(f_ss,false)),
fl AS (
  SELECT p.k,
    min(CASE WHEN f_verdict THEN snapshot END) t_verdict,
    min(CASE WHEN f_static  THEN snapshot END) t_static,
    min(CASE WHEN f_vt      THEN snapshot END) t_vt,
    min(CASE WHEN f_ss      THEN snapshot END) t_ss
  FROM p WHERE p.k IN (SELECT k FROM base) GROUP BY p.k)
SELECT f.*, b.snapshot AS t_first_seen FROM fl f JOIN base b USING (k)
WHERE t_verdict IS NOT NULL OR t_static IS NOT NULL OR t_vt IS NOT NULL OR t_ss IS NOT NULL""")
r = con.sql(f"""
SELECT count(*) later_flagged_any,
  sum(CASE WHEN t_verdict IS NOT NULL THEN 1 ELSE 0 END) later_verdict,
  sum(CASE WHEN t_static IS NOT NULL THEN 1 ELSE 0 END) later_static,
  sum(CASE WHEN t_vt IS NOT NULL THEN 1 ELSE 0 END) later_vt,
  sum(CASE WHEN t_ss IS NOT NULL THEN 1 ELSE 0 END) later_ss,
  sum(CASE WHEN t_verdict IS NOT NULL AND (t_vt<t_verdict OR t_static<t_verdict OR t_ss<t_verdict) THEN 1 ELSE 0 END) scanner_before_verdict,
  sum(CASE WHEN t_verdict IS NOT NULL AND coalesce(least(t_vt,t_static,t_ss)>t_verdict, true) THEN 1 ELSE 0 END) verdict_first_or_alone
FROM read_csv('{os.path.join(OUT, "rq2_first_flag.csv")}', all_varchar=true)""").df().iloc[0].to_dict()
summary["rq2"] = {k: int(v) for k, v in r.items()}

# ---------------- RQ3: agreement per snapshot ----------------
save("rq3_agreement", """
SELECT snapshot, count(*) n,
  sum(f_static::INT) static_pos, sum(f_vt::INT) vt_pos, sum(f_ss::INT) ss_pos, sum(f_verdict::INT) verdict_pos,
  sum((f_static AND f_vt)::INT)::DOUBLE / nullif(sum((f_static OR f_vt)::INT),0)  AS jaccard_static_vt,
  sum((f_static AND f_ss)::INT)::DOUBLE / nullif(sum((f_static OR f_ss)::INT),0)  AS jaccard_static_ss,
  sum((f_vt AND f_ss)::INT)::DOUBLE     / nullif(sum((f_vt OR f_ss)::INT),0)      AS jaccard_vt_ss,
  sum((f_static AND f_vt AND f_ss)::INT)::DOUBLE / nullif(sum((f_static OR f_vt OR f_ss)::INT),0) AS all_three_share,
  sum(((f_static::INT + f_vt::INT + f_ss::INT)=1)::INT)::DOUBLE / nullif(sum((f_static OR f_vt OR f_ss)::INT),0) AS single_scanner_share
FROM (SELECT snapshot, coalesce(f_static,false) f_static, coalesce(f_vt,false) f_vt,
             coalesce(f_ss,false) f_ss, coalesce(f_verdict,false) f_verdict FROM p)
GROUP BY 1 ORDER BY 1""")

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
for f in ("rq1_verdict_transitions", "rq1_model_transitions", "rq3_agreement", "rq1_changes_by_snapshot"):
    print(f"\n== {f}")
    print(open(os.path.join(OUT, f + ".csv")).read()[:3000])
