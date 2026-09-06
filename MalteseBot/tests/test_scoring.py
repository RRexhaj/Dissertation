"""Unit tests for the evaluation scoring functions (June harness + resubmission additions)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import has_expected_citation, has_invalid_citation, keyword_precision, rouge_l  # noqa: E402
from evaluate_multi import is_refusal, keyword_precision_alias, mirrors_language  # noqa: E402


# --- strict matcher (pre-registered, unchanged from June) -------------------

def test_strict_precision_full_match():
    ans = "In a built-up area the limit is 50 km/h unless signed otherwise."
    assert keyword_precision(ans, ["50 km/h", "built-up"]) == 1.0


def test_strict_precision_is_case_and_space_insensitive():
    assert keyword_precision("Limit: 50KM/H in BUILT-UP zones", ["50 km/h", "built-up"]) == 1.0


def test_strict_precision_misses_synonyms():
    # The documented weakness: a correct answer in different wording scores zero.
    assert keyword_precision("The limit is 50 milligrams per 100 ml of blood.", ["50 mg"]) == 0.0


def test_strict_precision_empty_facts():
    assert keyword_precision("anything", []) == 0.0


# --- alias matcher (resubmission) -------------------------------------------

def test_alias_precision_accepts_equivalent_form():
    aliases = {"50 mg": ["50 milligrams", "50mg"]}
    ans = "The limit is 50 milligrams per 100 ml of blood."
    assert keyword_precision_alias(ans, ["50 mg"], aliases) == 1.0


def test_alias_precision_never_below_strict():
    ans = "Refusing is an offence treated the same as exceeding the limit."
    facts = ["offence", "equivalent"]
    aliases = {"equivalent": ["treated the same"]}
    assert keyword_precision(ans, facts) == 0.5
    assert keyword_precision_alias(ans, facts, aliases) == 1.0
    assert keyword_precision_alias(ans, facts, None) == keyword_precision(ans, facts)


def test_alias_precision_partial():
    aliases = {"no motorway": ["does not have motorways"]}
    ans = "The open-road limit is 80 km/h."
    assert keyword_precision_alias(ans, ["80 km/h", "no motorway"], aliases) == 0.5


# --- citations -----------------------------------------------------------------

def test_expected_citation_present_and_absent():
    assert has_expected_citation("The limit is 50 km/h [S.L. 65.11 reg. 127].", ["S.L. 65.11"])
    assert not has_expected_citation("The limit is 50 km/h.", ["S.L. 65.11"])


def test_invalid_citation_detection():
    assert not has_invalid_citation("Limit 50 mg [Cap. 65 art. 15(2)(a)].")
    assert has_invalid_citation("Limit 50 mg [Cap. 99 art. 1].")
    assert not has_invalid_citation("No brackets here at all.")


# --- ROUGE-L ------------------------------------------------------------------

def test_rouge_l_identity_and_disjoint():
    s = "the fine doubles after fifteen days"
    assert rouge_l(s, s) == 1.0
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


# --- refusal + language mirroring (resubmission) ------------------------------

def test_refusal_detected_in_english_and_maltese():
    assert is_refusal("I don't have that information in Cap. 65. Contact LESA or a licensed advocate.")
    assert is_refusal("Ma nsibx informazzjoni dwar dan f'Kap. 65. Ċempel lil-LESA.")


def test_disclaimer_alone_is_not_a_refusal():
    ans = ("The limit is 50 km/h [S.L. 65.11 reg. 127]. AI-generated, not legal advice. "
           "For your specific case contact LESA +356 2122 2253.")
    assert not is_refusal(ans)


def test_language_mirroring():
    assert mirrors_language("Il-limitu huwa 50 km/s fil-bliet u l-irħula.", "mt")
    assert mirrors_language("The limit is 50 km/h in towns and villages.", "en")
    assert not mirrors_language("The limit is 50 km/h in towns and villages.", "mt")
