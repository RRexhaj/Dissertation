"""app.py — Dissertation user study (between-subjects design).

Each participant is randomly assigned to ONE condition on arrival:
  A - RAG + Transparency  (answer + inline citations + confidence label + disclaimer)
  B - RAG only            (same accurate answer, no citations, no confidence label)
  C - Baseline            (plain LLM answer, no retrieval, vaguer and uncited)

They rate 5 traffic-law scenarios under that single condition.
Trust is compared across the three groups using one-way ANOVA.

All answers are pre-generated so no API key is needed in deployment.

Run:
    streamlit run app.py
"""

import csv
import random
from datetime import datetime
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESPONSES_CSV = RESULTS_DIR / "user_study_responses.csv"

CONDITIONS = ["RAG + Transparency", "RAG (no transparency)", "Baseline"]

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": 1,
        "title": "Speed Limits",
        "question": "What is the speed limit in a built-up area in Malta, for example in Valletta?",
        "risk": "low",
    },
    {
        "id": 2,
        "title": "Unpaid Parking Fine",
        "question": "I received a parking ticket last week and have not paid it yet. What happens if I leave it unpaid for a month?",
        "risk": "low",
    },
    {
        "id": 3,
        "title": "Breathalyser Refusal",
        "question": "I was stopped by police and asked to take a breathalyser test. Can I legally refuse?",
        "risk": "medium",
    },
    {
        "id": 4,
        "title": "Demerit Points",
        "question": "I have accumulated 10 demerit points on my driving licence. How close am I to losing it?",
        "risk": "medium",
    },
    {
        "id": 5,
        "title": "Road Accident",
        "question": "I was involved in a minor car accident and the other driver wants to settle privately without involving insurance. What are my legal obligations?",
        "risk": "high",
    },
]

# ---------------------------------------------------------------------------
# Pre-generated answers — one set per condition
# ---------------------------------------------------------------------------

ANSWERS = {
    # ---- Scenario 1: Speed Limits ------------------------------------------
    1: {
        "RAG + Transparency": {
            "confidence": "High",
            "k_align": 3,
            "k_total": 3,
            "text": (
                "The default speed limit in a built-up area in Malta is **50 km/h** "
                "[S.L. 65.11 reg. 2]. This applies to urban roads including Valletta "
                "unless a road sign indicates a different limit [S.L. 65.11 reg. 2]. "
                "In school zones during school hours the limit drops to **30 km/h** "
                "[S.L. 65.11 reg. 3], and the same 30 km/h limit applies in "
                "designated residential zones [S.L. 65.11 reg. 3]. "
                "Speed limit signs always override the default [S.L. 65.11 reg. 2].\n\n"
                "*AI-generated, not legal advice. For your specific case contact "
                "LESA +356 2122 2253.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "The default speed limit in a built-up area in Malta is **50 km/h**. "
                "This applies to urban roads including Valletta unless a road sign "
                "indicates a different limit. In school zones during school hours "
                "the limit drops to **30 km/h**, and the same limit applies in "
                "designated residential zones. Speed limit signs always override "
                "the default."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta, the general speed limit in urban or built-up areas is "
                "around 50 km/h. In quieter residential streets or near schools it "
                "may be lower — sometimes 30 km/h — but this varies depending on "
                "the specific road. Road signs indicating a different limit always "
                "take priority over the general rule. It is always best to watch "
                "for posted signs as you drive."
            ),
        },
    },

    # ---- Scenario 2: Unpaid Parking Fine ------------------------------------
    2: {
        "RAG + Transparency": {
            "confidence": "High",
            "k_align": 3,
            "k_total": 3,
            "text": (
                "If you do not pay the fine within **15 days** of the date of issue, "
                "the amount doubles [Cap. 65 art. 15A]. If it remains unpaid after "
                "**30 days**, the fine triples [Cap. 65 art. 15A]. Since a week has "
                "already passed, you still have time to pay at the original amount "
                "if you act within the first 15 days. You can pay online at "
                "contraventions.gov.mt using the ticket reference number [Driver FAQ]. "
                "Continued non-payment may lead to referral to the Commissioner for "
                "Justice and potential court proceedings [Cap. 65 art. 15A].\n\n"
                "*AI-generated, not legal advice. For your specific case contact "
                "LESA +356 2122 2253.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "If you do not pay the fine within **15 days** of the date of issue, "
                "the amount doubles. If it remains unpaid after **30 days**, the fine "
                "triples. Since a week has already passed, you still have time to "
                "pay at the original amount if you act within the first 15 days. "
                "You can pay online at contraventions.gov.mt using your ticket "
                "reference number. Continued non-payment may lead to referral to "
                "the courts."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta, unpaid traffic fines typically increase if not settled "
                "promptly. After a certain number of days — often around two weeks — "
                "the fine can double, and after a further period it may increase "
                "again. If left unpaid for a long time the authorities can pursue "
                "legal action. Since you received the ticket last week you may still "
                "be within the window to pay the original amount. It is best to check "
                "the notice itself for the exact deadline and payment instructions."
            ),
        },
    },

    # ---- Scenario 3: Breathalyser Refusal -----------------------------------
    3: {
        "RAG + Transparency": {
            "confidence": "High",
            "k_align": 3,
            "k_total": 3,
            "text": (
                "No — refusing a breathalyser test when lawfully required by a police "
                "officer is itself an offence in Malta [S.L. 65.23 reg. 3]. "
                "A refusal is treated as equivalent to failing the test: you will be "
                "processed as if your blood alcohol level exceeded the legal limit "
                "[S.L. 65.23 reg. 3]. This carries **5 demerit points** and "
                "potentially criminal charges [Cap. 65, Second Schedule]. "
                "The general legal blood alcohol limit is 80 mg per 100 ml of blood; "
                "for new and professional drivers it is 20 mg per 100 ml "
                "[S.L. 65.23 reg. 2].\n\n"
                "*AI-generated, not legal advice. For your specific case contact "
                "LESA +356 2122 2253.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "No — refusing a breathalyser test when lawfully required by a police "
                "officer is itself an offence in Malta. A refusal is treated as "
                "equivalent to failing the test: you will be processed as if your "
                "blood alcohol level exceeded the legal limit. This carries 5 demerit "
                "points and potentially criminal charges. The general legal blood "
                "alcohol limit is 80 mg per 100 ml of blood; for new and professional "
                "drivers it is 20 mg per 100 ml."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta, if a police officer lawfully requires you to take a "
                "breathalyser test, refusing can have serious legal consequences. "
                "The refusal is generally treated in a similar way to failing the "
                "test, meaning you could face fines and other penalties as if you "
                "had been over the limit. The exact consequences can vary, but it "
                "is generally advisable to comply with the test when lawfully "
                "required to do so."
            ),
        },
    },

    # ---- Scenario 4: Demerit Points -----------------------------------------
    4: {
        "RAG + Transparency": {
            "confidence": "High",
            "k_align": 3,
            "k_total": 3,
            "text": (
                "The licence suspension threshold for most drivers is **12 demerit "
                "points** accumulated within any rolling two-year period [Cap. 65 "
                "art. 15D]. With 10 points you are **2 points away** from an "
                "automatic first suspension of three months [Cap. 65 art. 15D]. "
                "If you are a **new driver** — meaning you have held your licence "
                "for less than two years — the threshold is only **6 points**, "
                "which you have already passed, and suspension proceedings would "
                "apply [Cap. 65 art. 15D]. After a suspension the demerit counter "
                "resets to zero [Cap. 65 art. 15E].\n\n"
                "*AI-generated, not legal advice. For your specific case contact "
                "LESA +356 2122 2253.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "The licence suspension threshold for most drivers is **12 demerit "
                "points** accumulated within any rolling two-year period. With 10 "
                "points you are **2 points away** from an automatic first suspension "
                "of three months. If you are a **new driver** — meaning you have "
                "held your licence for less than two years — the threshold is only "
                "**6 points**, which you have already passed. After a suspension "
                "the demerit counter resets to zero."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta, demerit points accumulate on your driving licence and "
                "reaching a certain threshold leads to suspension. The general limit "
                "before suspension is around 12 points, so with 10 you are getting "
                "quite close. New drivers typically face a lower threshold, which "
                "can be as low as 6 points. It is worth checking with Transport "
                "Malta or LESA to confirm exactly where you stand and when your "
                "existing points were incurred, since points expire after a "
                "certain period."
            ),
        },
    },

    # ---- Scenario 5: Road Accident ------------------------------------------
    5: {
        "RAG + Transparency": {
            "confidence": "High",
            "k_align": 3,
            "k_total": 3,
            "text": (
                "After any road accident you are **legally required** to stop "
                "immediately and exchange your name, address, vehicle details, and "
                "insurance information with all other parties involved [Cap. 65 "
                "art. 19]. You are not obliged to agree to a private settlement — "
                "notifying your insurance company is your right and is not an offence "
                "[Cap. 65 art. 19]. If any person is injured you must also notify "
                "the police without delay [Cap. 65 art. 19]. Failing to stop after "
                "an accident carries **5 demerit points** and can result in criminal "
                "prosecution [Cap. 65, Second Schedule].\n\n"
                "*AI-generated, not legal advice. For your specific case contact "
                "LESA +356 2122 2253.*"
            ),
        },
        "RAG (no transparency)": {
            "text": (
                "After any road accident you are **legally required** to stop "
                "immediately and exchange your name, address, vehicle details, and "
                "insurance information with all other parties involved. You are not "
                "obliged to agree to a private settlement — notifying your insurance "
                "company is your right. If any person is injured you must also notify "
                "the police without delay. Failing to stop after an accident carries "
                "5 demerit points and can result in criminal prosecution."
            ),
        },
        "Baseline": {
            "text": (
                "In Malta, after a road accident you are generally required to stop "
                "and exchange contact and insurance details with the other driver. "
                "While private settlements are sometimes done for very minor "
                "incidents, they carry risks — if a dispute arises later you may "
                "have limited recourse. You are not legally required to agree to "
                "settle privately and you are entitled to involve your insurance "
                "company. If anyone is injured, contacting the police is important. "
                "It is always advisable to document the scene with photos before "
                "leaving."
            ),
        },
    },
}

LIKERT_QUESTIONS = [
    ("trust_overall",      "I trust the accuracy of this answer."),
    ("trust_rely",         "I would rely on this information if I were in this situation."),
    ("trust_confident",    "I feel confident the information provided is correct."),
    ("trust_clear",        "The answer was clear and easy to understand."),
    ("trust_legal",        "I would feel comfortable using this information before contacting LESA or a lawyer."),
]

CSV_FIELDS = [
    "participant_id", "timestamp", "scenario_id", "scenario_title", "risk_level",
    "condition",
    "trust_overall", "trust_rely", "trust_confident", "trust_clear", "trust_legal",
    "open_comment",
]

LIKERT_LABELS = {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_response(row: dict):
    write_header = not RESPONSES_CSV.exists()
    with open(RESPONSES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def render_answer(scenario_id: int, condition: str):
    data = ANSWERS[scenario_id][condition]
    if condition == "RAG + Transparency":
        label = data["confidence"]
        k_a = data["k_align"]
        k_t = data["k_total"]
        colour = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(label, "⚪")
        st.info(f"{colour} **Confidence: {label}** — {k_a}/{k_t} sources align")
    st.markdown(data["text"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MalteseLegalBot — User Study", layout="centered")

# ---- Welcome / consent -----------------------------------------------------
if "started" not in st.session_state:
    st.title("MalteseLegalBot — User Study")
    st.markdown("""
Thank you for taking part in this study. You will be shown **5 short road-traffic
law scenarios** and for each one you will read an AI-generated answer and rate how
much you trust it using a simple 1–5 scale.

The study takes approximately **5–8 minutes**.

**Participation is voluntary and anonymous.** No personal data is collected.
Responses are used solely for academic research as part of a BSc dissertation
at MCAST.

By clicking **Start** you confirm you are 18 or over and consent to participate.
""")
    if st.button("Start", type="primary"):
        st.session_state.started = True
        st.session_state.condition = random.choice(CONDITIONS)
        st.session_state.participant_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        st.session_state.scenario_index = 0
        st.session_state.ratings = {}
        st.rerun()
    st.stop()

# ---- Main survey loop ------------------------------------------------------
condition = st.session_state.condition
idx = st.session_state.scenario_index

if idx < len(SCENARIOS):
    scenario = SCENARIOS[idx]
    total = len(SCENARIOS)

    st.progress((idx) / total, text=f"Scenario {idx + 1} of {total}")
    st.subheader(f"Scenario {idx + 1}: {scenario['title']}")
    st.markdown(f"**Question:** {scenario['question']}")
    st.divider()
    st.markdown("**Chatbot answer:**")
    render_answer(scenario["id"], condition)
    st.divider()

    st.markdown("**Please rate the following statements (1 = Strongly Disagree, 5 = Strongly Agree):**")

    ratings = {}
    valid = True
    for key, label in LIKERT_QUESTIONS:
        val = st.radio(
            label,
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"{x} — {LIKERT_LABELS[x]}",
            index=None,
            key=f"{scenario['id']}_{key}",
            horizontal=True,
        )
        ratings[key] = val
        if val is None:
            valid = False

    comment = st.text_area(
        "Any comments on this answer? (optional)",
        key=f"{scenario['id']}_comment",
        height=80,
    )

    if st.button("Next →", type="primary", disabled=not valid):
        save_response({
            "participant_id": st.session_state.participant_id,
            "timestamp":      datetime.utcnow().isoformat(),
            "scenario_id":    scenario["id"],
            "scenario_title": scenario["title"],
            "risk_level":     scenario["risk"],
            "condition":      condition,
            **ratings,
            "open_comment":   comment.strip(),
        })
        st.session_state.scenario_index += 1
        st.rerun()

    if not valid:
        st.caption("Please answer all five questions before continuing.")

# ---- Thank you -------------------------------------------------------------
else:
    st.title("Thank you!")
    st.success(
        "Your responses have been recorded. "
        "This study is part of a BSc dissertation at MCAST investigating AI-powered "
        "legal information tools for Maltese citizens."
    )
    st.markdown(
        "If you have any questions about the study, contact the researcher at "
        "rexhajrei@gmail.com."
    )
