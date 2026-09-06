"""clean.py — apply documented exclusions, write cleaned CSVs + a cleaning log.

Exclusion rules (each justified in the dissertation Chapter 4):
  1. TEST RUN  — participant 20260603153429259754: first submission of the day,
     ran on the early instrument (only row carrying the now-removed attention_check
     item), double-submitted the probe, and straightlined the neutral midpoint on
     4/5 rows. Operator test, not a participant.
  2. DROPOUTS  — three participants who abandoned before completing all conditions
     and the final page (1 or 2 scenario rows, no 'final' row).
  3. REVERSE-CODE STRAIGHTLINERS — two participants whose Transparency row is
     5,5,5,5,5,5,5,5: t_reliable=5 AND t_wary=5 ("very reliable" yet "very
     suspicious") is internally contradictory and fails the embedded reverse-code
     validity check.
  4. DOUBLE-SUBMIT — one genuine participant submitted an identical 'speed' row
     twice 5 s apart; keep the first, drop the duplicate.

Outputs three datasets:
  cleaned_primary.csv          N=20  (pooled; rules 1-4)            -> PRIMARY
  cleaned_final_version.csv    N=19  (primary minus the one early-order participant)
  cleaned_keep_straightliners.csv N=22 (primary + the 2 straightliners back)
"""
import pandas as pd

CSV = r"C:\Users\rexha\Downloads\MalteseLegalBot Responses - Responses.csv"
OUT = r"C:\Users\rexha\Downloads\Dissertation_repo"
ITEMS = ["t_reliable","t_accurate","t_confident","t_wary","t_source","t_understand","t_use","t_rely"]

TEST_RUN        = "20260603153429259754"
DROPOUTS        = ["20260603162050977292", "20260603185257291635", "20260603185752650578"]
STRAIGHTLINERS  = ["20260603172759490593", "20260603192509523731"]
EARLY_ORDER     = "20260603154749602652"   # P2: transparency-shown-first version

df = pd.read_csv(CSV, dtype={"participant_id": str})
df.columns = [c.strip() for c in df.columns]

# rule 4: drop exact-duplicate scenario submissions, keep first
scen_key = ["participant_id","scenario_key","condition"] + ITEMS
is_scen = df["row_type"] == "scenario"
dup_mask = is_scen & df.duplicated(subset=scen_key, keep="first")
n_dup = int(dup_mask.sum())
df = df[~dup_mask].copy()

def drop_pids(d, pids):
    return d[~d["participant_id"].isin(pids)].copy()

# PRIMARY: rules 1-4, pooled across instrument versions
primary = drop_pids(df, [TEST_RUN] + DROPOUTS + STRAIGHTLINERS)
# sensitivity A: also drop the early-order participant (final-version-only)
final_only = drop_pids(primary, [EARLY_ORDER])
# sensitivity B: keep the straightliners in
keep_sl = drop_pids(df, [TEST_RUN] + DROPOUTS)

for name, d in [("cleaned_primary", primary),
                ("cleaned_final_version", final_only),
                ("cleaned_keep_straightliners", keep_sl)]:
    d.to_csv(f"{OUT}\\{name}.csv", index=False, encoding="utf-8")
    n = d[d["row_type"]=="scenario"]["participant_id"].nunique()
    print(f"{name}.csv -> {n} participants, {len(d)} rows")

print(f"\nDropped {n_dup} duplicate scenario row(s).")

# ---- correct manipulation-check tally from final rows (analyze.py mis-parses it) ----
print("\n=== MANIPULATION CHECK (final rows, PRIMARY sample) ===")
fin = primary[primary["row_type"]=="final"]
print("n final rows:", len(fin))
print("\nDid you notice sources/confidence differed?")
print(fin["manip_notice"].value_counts(dropna=False).to_string())
print("\nWhich did you trust most?")
print(fin["manip_most_trusted"].value_counts(dropna=False).to_string())
print("\nOpen reflections (non-empty):")
for r in fin["reflect_open"].dropna():
    if str(r).strip():
        print("  -", r)
