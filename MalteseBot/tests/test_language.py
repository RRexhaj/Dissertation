"""Tests for the bilingual language detector (bug found during the resubmission test pass)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance import detect_language  # noqa: E402


def test_short_english_query():
    assert detect_language("What is the speed limit in a built-up area in Malta?") == "en"


def test_long_english_answer_with_limit_and_this_is_english():
    # Regression: the June substring heuristic returned "mt" for this text
    # because "li" occurs in "limit" and "hi" in "this".
    ans = ("The speed limit in a built-up area is 50 km/h unless a sign indicates otherwise "
           "[S.L. 65.11 reg. 127]. This limit applies in towns and villages. AI-generated, not "
           "legal advice. For your specific case contact LESA +356 2122 2253.")
    assert detect_language(ans) == "en"


def test_maltese_with_diacritics():
    assert detect_language("X'inhu l-limitu tal-veloċità f'żona urbana f'Malta?") == "mt"


def test_maltese_without_diacritics_uses_function_words():
    assert detect_language("Kemm nista nsuq malajr fuq triq miftuha u hemm limitu?") == "mt"


def test_maltese_answer():
    ans = ("Il-limitu tal-velocita f zona urbana huwa 50 km/s sakemm ma jkunx indikat mod iehor "
           "[S.L. 65.11 reg. 127]. Iggenerat mill-IA, mhux parir legali.")
    assert detect_language(ans) == "mt"
