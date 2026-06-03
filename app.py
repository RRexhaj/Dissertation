"""app.py — Dissertation user study (within-subjects, English).

Research design
---------------
WITHIN-SUBJECTS. Every participant experiences all three presentation
conditions, each tied to a different scenario (so the same answer is never shown
twice with/without transparency, which would cue the hypothesis):

  A - RAG + Transparency   answer + inline citations + confidence label
                           + a Sources panel + disclaimer
  B - RAG (no transparency) the same accurate answer, but as a plain "standard
                           chatbot" reply: no citations, no confidence, no sources
  C - Baseline             a vaguer, uncited general-knowledge "standard chatbot"
                           reply

Three "rotated" scenarios (speed / alcohol / penalty points) are assigned one
each to A, B, C, with the scenario->condition mapping randomised per participant.
A fourth scenario is a CALIBRATION PROBE: an honestly LOW-confidence, uncited
answer shown in the Transparency condition, used to test whether transparency
helps users *appropriately* lower their trust (calibrated trust; Lee & See, 2004)
rather than just raising it.

Presentation order: the two "standard chatbot" answers (no sources/confidence)
are shown FIRST, then the two answers that carry a confidence score + sources.
This makes the contrast between a regular chatbot and the transparent assistant
explicit. (Note for write-up: this fixed grouping trades counterbalancing of
presentation order for a clearer manipulation; scenario<->condition mapping is
still randomised.)

Trust is measured with a HYBRID scale per scenario (1-5 Likert):
  * trust core adapted from Jian, Bisantz & Drury (2000) trust-in-automation
    (reliable / accurate / confident / suspicious[reverse-coded]),
  * behavioural-intention items adapted from Davis (1989) TAM (use / rely),
  * two legal-context items (source transparency; comprehension of rights).

All answers are pre-generated and factually verified against the Maltese statutes
(Cap. 65; S.L. 65.11 reg. 127; Cap. 65 arts 15-15G; S.L. 65.18), so no API key is
needed in deployment.

Run:
    streamlit run app.py
"""

import csv
import random
import re
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESPONSES_CSV = RESULTS_DIR / "user_study_responses.csv"

CONDITIONS = ["RAG + Transparency", "RAG (no transparency)", "Baseline"]
ROTATED = ["speed", "alcohol", "points"]   # one each -> A / B / C
PROBE = "probe"                            # always Transparency, LOW confidence
RESEARCHER_EMAIL = "rexhajrei@gmail.com"
LESA = "+356 2122 2253"

# ---------------------------------------------------------------------------
# UI strings
# ---------------------------------------------------------------------------

T = {
    "title": "MalteseLegalBot — Research Study",
    "consent_md": """
Thank you for taking part in this study, run as part of a **BSc dissertation at MCAST**
on AI-powered legal-information tools for Maltese road-traffic law.

You will read **four short traffic-law scenarios**. For each one you will see an
AI-generated answer and rate how much you trust it on a 1–5 scale. It takes about
**6–8 minutes**.

- Participation is **voluntary** — you may stop at any time by closing the page.
- Responses are **anonymous**; no name, email or IP address is stored.
- Data is used **only** for academic research and reported in aggregate.
- The answers shown are AI-generated and are **not legal advice**.

By clicking **Start** you confirm you are **18 or over** and consent to take part.
""",
    "start": "Start",
    "demo_title": "A few questions about you",
    "demo_caption": "This helps describe who took part.",
    "age": "Your age group",
    "gender": "Gender",
    "education": "Highest level of education completed",
    "ai_use": "How often do you use AI chatbots (e.g. ChatGPT)?",
    "law_fam": "How familiar are you with Maltese road-traffic law? (1 = not at all, 5 = very familiar)",
    "native": "Your first language",
    "continue": "Continue",
    "scenario": "Scenario",
    "of": "of",
    "question": "Question",
    "bot_answer": "Chatbot answer",
    "standard_header": "🤖 Standard chatbot answer",
    "standard_caption": "A typical AI chatbot reply — no sources or confidence indicator shown.",
    "rate_intro": "Please rate each statement (1 = Strongly Disagree, 5 = Strongly Agree):",
    "comment": "Any comments on this answer? (optional)",
    "next": "Next →",
    "answer_all": "Please answer every statement before continuing.",
    "sources": "📄 Sources",
    "confidence": "Confidence",
    "sources_align": "sources align",
    "page_disclaimer": "Part of a research study · answers are AI-generated · not legal advice.",
    "final_title": "Almost done — two last questions",
    "manip_q": "During the study, some answers showed their sources and a confidence level, while others did not. Did you notice this difference?",
    "manip_most": "Overall, which kind of answer did you trust the most?",
    "reflect": "In your own words, what made an answer feel trustworthy or untrustworthy to you? (optional)",
    "submit": "Submit",
    "done_title": "Thank you!",
    "done_md": (
        "Your responses have been recorded. This study investigated whether showing "
        "**sources and a confidence level** changes how much people trust an AI legal "
        "assistant — including whether it helps you trust a *less certain* answer less.\n\n"
        "Some answers deliberately showed low confidence to test this. For any real "
        f"traffic-law matter, please contact **LESA ({LESA})** or a licensed advocate.\n\n"
        f"Questions about the study? Contact the researcher at **{RESEARCHER_EMAIL}**."
    ),
}

LIKERT_LABELS = {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"}

CONF_LABEL = {"High": "High", "Medium": "Medium", "Low": "Low"}
CONF_COLOUR = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

# ---------------------------------------------------------------------------
# Trust scale (hybrid). t_wary is REVERSE-coded in analysis.
# ---------------------------------------------------------------------------

TRUST_ITEMS = ["t_reliable", "t_accurate", "t_confident", "t_wary",
               "t_source", "t_understand", "t_use", "t_rely"]

TRUST_LABELS = {
    "t_reliable":   "This answer is reliable.",
    "t_accurate":   "I trust the accuracy of this answer.",
    "t_confident":  "I would feel confident acting on this answer.",
    "t_wary":       "I am suspicious of this answer.",
    "t_source":     "The answer made it clear where the information comes from.",
    "t_understand": "After reading this, I understand my legal rights or obligations in this situation.",
    "t_use":        "I would use a tool like this for my own traffic-law questions.",
    "t_rely":       "I would feel comfortable relying on this answer before contacting LESA or a lawyer.",
}

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

AGE_BANDS = ["18–24", "25–34", "35–44", "45–54", "55+"]

DEMO_OPTIONS = {
    "gender": ["Female", "Male", "Other", "Prefer not to say"],
    "education": ["Secondary or below", "Post-secondary / MCAST / Sixth Form",
                  "Bachelor's degree", "Master's or higher", "Prefer not to say"],
    "ai_use_freq": ["Never", "Rarely", "Monthly", "Weekly", "Daily"],
    "native_language": ["Maltese", "English", "Both equally", "Other"],
}

MANIP_NOTICE = ["Yes, clearly", "Yes, somewhat", "No, I did not notice"]
MANIP_MOST = ["Answers that showed sources and confidence",
              "Answers without sources or confidence",
              "I trusted them about the same"]

# ---------------------------------------------------------------------------
# Scenarios (verified against Maltese statutes)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "speed": {
        "risk": "low",
        "title": "Speed Limits",
        "question": "What is the speed limit in a built-up area in Malta, for example in Valletta?",
    },
    "alcohol": {
        "risk": "high",
        "title": "Breath Test & Alcohol Limit",
        "question": "A police officer asked me to take a breathalyser test. Can I refuse, and what is the legal alcohol limit?",
    },
    "points": {
        "risk": "medium",
        "title": "Penalty Points",
        "question": "I have 10 penalty points on my driving licence. How close am I to losing it?",
    },
    "probe": {
        "risk": "medium",
        "title": "Lending Your Car",
        "question": "If I lend my car to a friend and they are caught speeding, who gets the penalty points — me or them?",
    },
}

# ---------------------------------------------------------------------------
# Answers — one set per condition.
# Transparency text carries inline [citations] + a closing disclaimer.
# ---------------------------------------------------------------------------

ANSWERS = {
    "speed": {
        "RAG + Transparency": {
            "confidence": "High", "k_align": 3, "k_total": 3,
            "text": (
                "In Malta the default speed limit in a built-up area (towns and "
                "villages) is **50 km/h** [S.L. 65.11 reg. 127]. On roads outside "
                "built-up areas the default is **80 km/h** [S.L. 65.11 reg. 127] — "
                "Malta has no motorways with a higher limit. A lower limit applies "
                "wherever a road sign indicates one, and signs always override the "
                "default [S.L. 65.11 reg. 127].\n\n"
                f"*AI-generated, not legal advice. For your specific situation contact LESA on {LESA}.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "In Malta the default speed limit in a built-up area (towns and "
                "villages) is 50 km/h. On roads outside built-up areas the default "
                "is 80 km/h — Malta has no motorways with a higher limit. A lower "
                "limit applies wherever a road sign indicates one, and signs always "
                "override the default."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta the speed limit in urban or built-up areas is generally "
                "around 50 km/h. On wider or rural roads it is higher, and on some "
                "stretches it can be more. Limits can vary from road to road, so the "
                "safest thing is to follow the posted signs, which take priority over "
                "any general rule."
            ),
        },
    },

    "alcohol": {
        "RAG + Transparency": {
            "confidence": "High", "k_align": 3, "k_total": 3,
            "text": (
                "Refusing a breath test when a police officer lawfully requires one "
                "is itself an offence, and a refusal is treated as failing the test "
                "[Cap. 65 art. 15]. The general legal limit is **50 mg of alcohol per "
                "100 ml of blood (0.05% BAC)** [Cap. 65 art. 15]. For new/probationary "
                "and professional drivers the limit is lower, **20 mg per 100 ml "
                "(0.02%)** [Cap. 65 art. 15], and for bus drivers it is effectively "
                "zero [Cap. 65 art. 15].\n\n"
                f"*AI-generated, not legal advice. For your specific situation contact LESA on {LESA}.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "Refusing a breath test when a police officer lawfully requires one "
                "is itself an offence, and a refusal is treated as failing the test. "
                "The general legal limit is 50 mg of alcohol per 100 ml of blood "
                "(0.05% BAC). For new/probationary and professional drivers the limit "
                "is lower, 20 mg per 100 ml (0.02%), and for bus drivers it is "
                "effectively zero."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta you are generally expected to take a breathalyser test when "
                "the police ask, and refusing can be treated much like failing it, "
                "with serious penalties. There is a legal blood-alcohol limit and it "
                "is lower for newer or professional drivers, but the exact figures "
                "can vary, so it is best to avoid driving after drinking and to "
                "comply if you are asked to be tested."
            ),
        },
    },

    "points": {
        "RAG + Transparency": {
            "confidence": "High", "k_align": 3, "k_total": 3,
            "text": (
                "A driver who reaches **12 penalty points within a 12-month period** "
                "has their driving licence revoked [S.L. 65.18]. With 10 points you "
                "are **2 points away** from that threshold [S.L. 65.18]. The same "
                "12-point threshold applies to probationary (new) drivers during their "
                "licence probation — there is no separate lower limit [S.L. 65.18]. "
                "After a first revocation a driver may apply to be re-licensed after "
                "**2 months** [S.L. 65.18].\n\n"
                f"*AI-generated, not legal advice. For your specific situation contact LESA on {LESA}.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "A driver who reaches 12 penalty points within a 12-month period has "
                "their driving licence revoked. With 10 points you are 2 points away "
                "from that threshold. The same 12-point threshold applies to "
                "probationary (new) drivers during their licence probation — there is "
                "no separate lower limit. After a first revocation a driver may apply "
                "to be re-licensed after 2 months."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta penalty points build up on your licence and once you reach "
                "a certain total your licence can be taken away. The usual limit is "
                "around twelve points, so with ten you are getting fairly close. New "
                "drivers may be treated more strictly. It is worth checking your exact "
                "standing with the authorities, as the rules on how long points stay "
                "on your record can vary."
            ),
        },
    },

    # ---- Calibration probe (Transparency, honestly LOW confidence) ---------
    "probe": {
        "RAG + Transparency": {
            "confidence": "Low", "k_align": 0, "k_total": 3,
            "text": (
                "I'm not fully certain about this one. As a general principle penalty "
                "points apply to the driver who actually committed the offence rather "
                "than to the vehicle's owner — but exactly how an offence is attributed "
                "when someone else was driving, and what the owner must do to identify "
                "that driver, is not something I can confirm clearly from the "
                "legislation I have access to. Please check this directly with LESA "
                "before relying on it.\n\n"
                f"*AI-generated, not legal advice. For your specific situation contact LESA on {LESA}.*"
            ),
        },
    },
}

CSV_FIELDS = [
    "participant_id", "timestamp", "language", "row_type",
    "age_band", "gender", "education", "ai_use_freq", "law_familiarity", "native_language",
    "order_position", "scenario_key", "scenario_title", "risk_level",
    "condition", "confidence_shown", "is_probe",
    "t_reliable", "t_accurate", "t_confident", "t_wary",
    "t_source", "t_understand", "t_use", "t_rely",
    "manip_notice", "manip_most_trusted", "reflect_open",
    "comment",
]


# ---------------------------------------------------------------------------
# Persistence — Google Sheets webhook (DO NOT alter logic)
# ---------------------------------------------------------------------------

def _webhook_url() -> str:
    """Return the Apps Script web-app URL from secrets, or '' if not configured."""
    try:
        return str(st.secrets["webhook_url"]).strip()
    except Exception:
        return ""


def save_response(row: dict):
    """Persist one survey row.

    On a DEPLOYED app the local CSV lives on Streamlit Cloud's ephemeral disk and
    is wiped on every restart/sleep, so the Google Sheet (written via the Apps
    Script webhook in secrets) is the real, durable store. The local CSV is kept
    as a convenience backup for runs on your own machine.
    """
    full = {k: row.get(k, "") for k in CSV_FIELDS}
    # 1. Local CSV backup — authoritative only when running locally.
    write_header = not RESPONSES_CSV.exists()
    with open(RESPONSES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(full)

    # 2. Remote persistence — required for the deployed study.
    url = _webhook_url()
    if url:
        try:
            resp = requests.post(
                url, json={"fields": CSV_FIELDS, "row": full}, timeout=10
            )
            resp.raise_for_status()
        except Exception as e:
            # Don't lose the participant: surface the problem so it gets noticed.
            st.warning(
                "Your answer was saved, but syncing to the research database had a "
                "hiccup. You can continue — but if you are the researcher, check the "
                f"webhook. ({e})"
            )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def build_plan() -> list[dict]:
    """One participant's scenario plan.

    Scenario<->condition mapping is randomised, but the two 'standard chatbot'
    answers (Baseline + no-transparency) are presented FIRST, then the two
    answers carrying a confidence score + sources (Transparency + probe)."""
    conds = CONDITIONS[:]
    random.shuffle(conds)
    rotated = [{"key": k, "condition": c} for k, c in zip(ROTATED, conds)]
    items = rotated + [{"key": PROBE, "condition": "RAG + Transparency"}]

    standard = [x for x in items if x["condition"] != "RAG + Transparency"]
    transparent = [x for x in items if x["condition"] == "RAG + Transparency"]
    random.shuffle(standard)
    random.shuffle(transparent)
    return standard + transparent


def render_answer(key: str, condition: str):
    data = ANSWERS[key][condition]
    text = data["text"]
    if condition == "RAG + Transparency":
        label = data["confidence"]
        a, t = data["k_align"], data["k_total"]
        st.info(
            f"{CONF_COLOUR.get(label, '⚪')} **{T['confidence']}: "
            f"{CONF_LABEL[label]}** — {a}/{t} {T['sources_align']}"
        )
        st.markdown(text)
        cites = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", text)))
        if cites:
            st.markdown(f"**{T['sources']}**")
            for c in cites:
                st.markdown(f"- `{c}`")
    else:
        # Make it explicit that this is a plain, regular chatbot reply.
        st.markdown(f"**{T['standard_header']}**")
        st.caption(T["standard_caption"])
        st.markdown(text)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MalteseLegalBot — User Study", layout="centered")

ss = st.session_state
if "stage" not in ss:
    ss.stage = "consent"

# ---- Stage 1: consent -------------------------------------------------------
if ss.stage == "consent":
    st.title(T["title"])
    st.markdown(T["consent_md"])
    if st.button(T["start"], type="primary"):
        ss.participant_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        ss.plan = build_plan()
        ss.idx = 0
        ss.demo = {}
        ss.stage = "demographics"
        st.rerun()
    st.stop()

# ---- Stage 2: demographics --------------------------------------------------
if ss.stage == "demographics":
    st.title(T["demo_title"])
    st.caption(T["demo_caption"])

    age = st.radio(T["age"], AGE_BANDS, index=None, horizontal=True, key="d_age")
    gender = st.radio(T["gender"], DEMO_OPTIONS["gender"], index=None, key="d_gender")
    edu = st.radio(T["education"], DEMO_OPTIONS["education"], index=None, key="d_edu")
    ai = st.radio(T["ai_use"], DEMO_OPTIONS["ai_use_freq"], index=None, horizontal=True, key="d_ai")
    fam = st.slider(T["law_fam"], 1, 5, 3, key="d_fam")
    native = st.radio(T["native"], DEMO_OPTIONS["native_language"], index=None, horizontal=True, key="d_native")

    ready = all(v is not None for v in (age, gender, edu, ai, native))
    if st.button(T["continue"], type="primary", disabled=not ready):
        ss.demo = {
            "age_band": age, "gender": gender, "education": edu,
            "ai_use_freq": ai, "law_familiarity": fam, "native_language": native,
        }
        ss.stage = "survey"
        st.rerun()
    if not ready:
        st.caption(T["answer_all"])
    st.stop()

# ---- Stage 3: survey loop ---------------------------------------------------
if ss.stage == "survey":
    plan = ss.plan
    idx = ss.idx
    total = len(plan)

    if idx >= total:
        ss.stage = "final"
        st.rerun()

    step = plan[idx]
    key, condition = step["key"], step["condition"]
    sc = SCENARIOS[key]
    is_probe = key == PROBE

    st.progress(idx / total, text=f"{T['scenario']} {idx + 1} {T['of']} {total}")
    st.subheader(f"{T['scenario']} {idx + 1}: {sc['title']}")
    st.markdown(f"**{T['question']}:** {sc['question']}")
    st.divider()
    st.markdown(f"**{T['bot_answer']}:**")
    render_answer(key, condition)
    st.divider()

    st.markdown(f"**{T['rate_intro']}**")
    ratings = {}
    valid = True
    for item in TRUST_ITEMS:
        val = st.radio(
            TRUST_LABELS[item],
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"{x} — {LIKERT_LABELS[x]}",
            index=None, horizontal=True, key=f"{idx}_{item}",
        )
        ratings[item] = val
        if val is None:
            valid = False

    comment = st.text_area(T["comment"], key=f"{idx}_comment", height=80)

    if st.button(T["next"], type="primary", disabled=not valid):
        save_response({
            "participant_id": ss.participant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "language": "en",
            "row_type": "scenario",
            **ss.demo,
            "order_position": idx + 1,
            "scenario_key": key,
            "scenario_title": sc["title"],
            "risk_level": sc["risk"],
            "condition": condition,
            "confidence_shown": ANSWERS[key][condition].get("confidence", ""),
            "is_probe": int(is_probe),
            **ratings,
            "comment": comment.strip(),
        })
        ss.idx += 1
        st.rerun()

    if not valid:
        st.caption(T["answer_all"])
    st.caption(f"🔒 {T['page_disclaimer']}")
    st.stop()

# ---- Stage 4: final questions ----------------------------------------------
if ss.stage == "final":
    st.title(T["final_title"])
    notice = st.radio(T["manip_q"], MANIP_NOTICE, index=None, key="f_notice")
    most = st.radio(T["manip_most"], MANIP_MOST, index=None, key="f_most")
    reflect = st.text_area(T["reflect"], key="f_reflect", height=120)

    ready = notice is not None and most is not None
    if st.button(T["submit"], type="primary", disabled=not ready):
        save_response({
            "participant_id": ss.participant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "language": "en",
            "row_type": "final",
            **ss.demo,
            "manip_notice": notice,
            "manip_most_trusted": most,
            "reflect_open": reflect.strip(),
        })
        ss.stage = "done"
        st.rerun()
    if not ready:
        st.caption(T["answer_all"])
    st.stop()

# ---- Stage 5: debrief -------------------------------------------------------
st.title(T["done_title"])
st.success(T["done_md"])
