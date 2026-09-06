# GPT Powered Maltese Legal Bot for Traffic Offences

BSc (Hons) Software Development dissertation, MCAST Institute of ICT. Author: Rei Rexhaj. Supervisor: Ms Annalise Consoli.
This repository holds the prototype, the evaluation harness, the statistical analysis and the user-study instrument
referred to in the dissertation (September 2026 resubmission). Every table and figure in Chapter 4 is regenerated
from the files here.

## Layout

| Path | Contents |
|---|---|
| `MalteseBot/` | The prototype: `corpus.py` (hierarchical chunker), `retrieval.py` (hybrid BM25 + dense retrieval, cross-encoder rerank), `prompts.py` (constrained bilingual prompt), `compliance.py` (EU AI Act banner, pseudonymised logging, language detection), `app.py` (Streamlit UI), `maltese_legal_chatbot.py` (CLI), `data/cleaned_txt/` (validated bilingual corpus) |
| `MalteseBot/eval/` | `gold_queries.json` (24 statute-validated questions), `fact_aliases.json`, `oos_queries.json`, `evaluate.py` (June 2026 harness, unchanged), `evaluate_multi.py` (multi-model harness), `ablation.py`, `llm.py` (model adapter), `stats.py`, `figures.py`, `make_rating_sheet.py`, `judge_agreement.py`, `make_appendix_tables.py`, `prices.json`, results (`results_*.csv`, `summary_*.json`, `stats_summary.json`), `statutes/` (consolidated statute text used for validation), `tables/` (LaTeX fragments) |
| `MalteseBot/tests/` | pytest suite (27 tests: scoring, corpus, retrieval, language detection) |
| `app.py` | The user-study instrument (Streamlit), deployed at https://dissertation.streamlit.app/ |
| `clean.py`, `cleaned_primary.csv` | Screening of the raw survey export and the analytic dataset (N = 20; pseudonymous IDs only) |
| `analyze.py`, `analyze_v2.py` | User-study statistics (Friedman, Wilcoxon, Cronbach's alpha, effect sizes, order and scenario checks) |
| `figures/`, `tables/` | User-study figures and LaTeX table fragments |

## Setup

```bash
cd MalteseBot
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install pytest openpyxl
echo OPENAI_API_KEY=sk-... > .env
```

## Reproduce

```bash
# from MalteseBot/
python -m pytest tests -q                       # offline test suite
python eval/evaluate_multi.py --smoke           # 1 question x 6 models, sanity check
python eval/evaluate_multi.py --runs 3          # 2 conditions x 6 models x 24 questions x 3 runs (resumable)
python eval/evaluate_multi.py --oos             # out-of-scope robustness set
python eval/ablation.py --downstream            # retrieval ablation + downstream GPT-4o answers
python eval/stats.py                            # stats_summary.json + eval/tables/*.tex
python eval/figures.py --out ../Figures         # Chapter 4 figures
python eval/make_appendix_tables.py             # gold set and per-question appendix tables
python eval/make_rating_sheet.py                # blind human-rating workbook (+ hidden key)
python eval/judge_agreement.py                  # after the workbook is filled in: Cohen's kappa etc.

# from the repository root
python analyze_v2.py                            # user-study statistics, tables and figures
```

Run the chatbot itself with `streamlit run app.py` from `MalteseBot/` (first answer takes about a minute while the
cross-encoder loads). The gold set, aliases and out-of-scope questions are never modified by any script.

## Notes

* Model snapshots, decoding parameters and list prices are recorded in `eval/prices.json` and in every results row.
* The June 2026 results (`eval/results_per_query.csv`, `eval/summary.json`) are kept unchanged as the record of the
  first evaluation cycle.
* No user data leaves this repository beyond the pseudonymous survey CSVs; the `.env` key is never committed.
