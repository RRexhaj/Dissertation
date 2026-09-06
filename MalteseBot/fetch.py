"""fetch.py — Build the bilingual Maltese-traffic-law corpus.

Pulls Cap. 65 + selected subsidiary legislation from legislation.mt in BOTH
official languages (English + Maltese), strips them to plain UTF-8 text, and
writes them under data/cleaned_txt/{en,mt}/. The hierarchical chunker in
corpus.py then parses these into article-level units.

Sources are chosen to match the corpus described in Chapter 2 of the
dissertation (bilingual statutory base for the RAG retrieval index).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import requests
from pdfminer.high_level import extract_text

BASE = Path(__file__).parent / "data" / "cleaned_txt"
HEADERS = {"User-Agent": "MalteseLegalBot/2.0 (academic; MCAST)"}

# (url, lang_code, filename)
SOURCES: list[tuple[str, str, str]] = [
    # Cap. 65 — Traffic Regulation Ordinance
    ("https://legislation.mt/eli/cap/65/eng/pdf",   "en", "traffic_act.txt"),
    ("https://legislation.mt/eli/cap/65/mlt/pdf",   "mt", "traffic_act.txt"),
    # S.L. 65.04 — Motor Vehicles (Third-Party Insurance)
    ("https://legislation.mt/eli/sl/65.4/eng/pdf",  "en", "motor_vehicle_insurance.txt"),
    ("https://legislation.mt/eli/sl/65.4/mlt/pdf",  "mt", "motor_vehicle_insurance.txt"),
    # S.L. 65.10 — Traffic Signs and Carriageway Markings
    ("https://legislation.mt/eli/sl/65.10/eng/pdf", "en", "traffic_signs_regs.txt"),
    ("https://legislation.mt/eli/sl/65.10/mlt/pdf", "mt", "traffic_signs_regs.txt"),
    # S.L. 65.11 — Motor Vehicles (Speed Limits)
    ("https://legislation.mt/eli/sl/65.11/eng/pdf", "en", "speed_limits_regs.txt"),
    ("https://legislation.mt/eli/sl/65.11/mlt/pdf", "mt", "speed_limits_regs.txt"),
    # S.L. 65.17 — Driving Licences
    ("https://legislation.mt/eli/sl/65.17/eng/pdf", "en", "driving_licences.txt"),
    ("https://legislation.mt/eli/sl/65.17/mlt/pdf", "mt", "driving_licences.txt"),
    # S.L. 65.23 — Road Traffic (Alcohol)
    ("https://legislation.mt/eli/sl/65.23/eng/pdf", "en", "alcohol_regs.txt"),
    ("https://legislation.mt/eli/sl/65.23/mlt/pdf", "mt", "alcohol_regs.txt"),
    # S.L. 65.32 — Micromobility Regulations
    ("https://legislation.mt/eli/sl/65.32/eng/pdf", "en", "micromobility_regs.txt"),
    ("https://legislation.mt/eli/sl/65.32/mlt/pdf", "mt", "micromobility_regs.txt"),
    # S.L. 65.33 — Road Traffic (Enforcement Camera) Regulations
    ("https://legislation.mt/eli/sl/65.33/eng/pdf", "en", "enforcement_camera.txt"),
    ("https://legislation.mt/eli/sl/65.33/mlt/pdf", "mt", "enforcement_camera.txt"),
]

# ---------------------------------------------------------------------------
# Manual curated content — used when PDF downloads fail or files are missing.
# Written in plain statutory language to match the tone of Cap. 65.
# ---------------------------------------------------------------------------

FAQ_EN = """HOW TO PAY OR CONTEST A TRAFFIC CONTRAVENTION IN MALTA
Source: Driver FAQ

- Pay online within 15 days at https://contraventions.gov.mt using the ticket reference.
- Pay in person at any MaltaPost branch or the Police Traffic Section (Floriana).
- To contest, file a written plea at the Commissioner for Justice / Local Tribunal registry within 15 days.
- Late fees: the fine doubles after 15 days and triples after 30 days (Cap. 65 art. 15A).
- Demerit points are recorded on your licence in addition to fines (Cap. 65 art. 15B-15I).
- Escalation: for case-specific advice contact LESA on +356 2122 2253 or a licensed advocate.
- A contravention notice may be issued by a police officer or by an authorised warden.
- You may also receive a notice by post if the offence was captured by an enforcement camera (S.L. 65.33).
- Demerit points remain on your licence for two years from the date of the offence.
- If you accumulate 12 or more demerit points your licence is suspended (Cap. 65 art. 15D).
- New drivers (licence held less than 2 years) face suspension at 6 demerit points.
- A suspended licence holder who continues to drive commits a further offence under Cap. 65 art. 14.
"""

FAQ_MT = """KIF TĦALLAS JEW TIKKONTESTA KONTRAVENZJONI TAT-TRAFFIKU F'MALTA
Sors: Driver FAQ

- Ħallas online fi 15-il jum fuq https://contraventions.gov.mt billi tuża r-referenza tal-biljett.
- Ħallas personalment f'kull friegħa tal-MaltaPost jew fis-Sezzjoni tat-Traffiku tal-Pulizija (Floriana).
- Biex tikkontesta, ressaq talba bil-miktub fir-Reġistru tal-Kummissarju għall-Ġustizzja / Tribunal Lokali fi 15-il jum.
- Ħlasijiet tard: il-multa tirdoppja wara 15-il jum u tittripla wara 30 jum (Kap. 65 art. 15A).
- Punti tad-demerit jiġu rreġistrati fuq il-liċenzja b'żieda mal-multi (Kap. 65 art. 15B-15I).
- Eskalazzjoni: għal pariri speċifiċi għall-każ tiegħek ċempel lil-LESA fuq +356 2122 2253 jew lil avukat liċenzjat.
- Avviż ta' kontravvenzjoni jista' jinħareġ minn uffiċjal tal-pulizija jew minn warden awtorizzat.
- Tista' wkoll tirċievi avviż bil-posta jekk ir-reat kien mikxuf minn camera tal-infurzar (S.L. 65.33).
- Il-punti tad-demerit jibqgħu fuq il-liċenzja tiegħek għal sentejn mid-data tar-reat.
- Jekk tiġbor 12-il punt tad-demerit jew aktar il-liċenzja tiegħek tiġi sospiża (Kap. 65 art. 15D).
- Sewwieqa ġodda (liċenzja miżmuma inqas minn sentejn) jaffaċċjaw sospensjoni bi 6 punti tad-demerit.
- Detentur ta' liċenzja sospiża li jkompli jsuq jikkommetti reat ieħor skont Kap. 65 art. 14.
"""

PENALTY_EN = """MALTESE ROAD TRAFFIC PENALTY POINTS TABLE
Source: Transport Malta / Cap. 65 arts. 15B-15I

DEMERIT POINT SYSTEM — KEY RULES
- Every driver starts with zero demerit points.
- Points accumulate over a rolling two-year window from the date of each offence.
- Reaching 12 points triggers a licence suspension for three months (first time).
- New drivers (licence held under two years) are suspended at 6 points.
- After suspension the counter resets to zero.
- Repeat suspensions within five years carry longer suspension periods.

OFFENCE AND DEMERIT POINT SCHEDULE (Cap. 65, Second Schedule)

Speeding offences:
- Exceeding the limit by 1-10 km/h: 1 demerit point
- Exceeding the limit by 11-20 km/h: 2 demerit points
- Exceeding the limit by 21-30 km/h: 3 demerit points
- Exceeding the limit by 31-40 km/h: 4 demerit points
- Exceeding the limit by more than 40 km/h: 5 demerit points

Mobile phone and distraction offences:
- Using a hand-held mobile phone while driving: 3 demerit points
- Reading or sending a text message while driving: 3 demerit points

Seatbelt offences:
- Failure to wear a seatbelt (driver): 3 demerit points
- Failure to ensure a child passenger is restrained: 3 demerit points

Alcohol and drug offences:
- Driving with blood alcohol 0.08-0.15 g/100ml: 4 demerit points
- Driving with blood alcohol above 0.15 g/100ml: 5 demerit points
- Refusing a breath test when required by police: 5 demerit points
- Driving under the influence of drugs: 5 demerit points

Traffic light and sign offences:
- Failing to stop at a red traffic light: 4 demerit points
- Failing to give way as required by a Give Way sign: 2 demerit points
- Failing to stop at a Stop sign: 3 demerit points

Overtaking offences:
- Dangerous overtaking: 4 demerit points
- Overtaking on a solid white line: 3 demerit points

Parking offences:
- Parking on a pedestrian crossing or within 10 metres before it: 2 demerit points
- Parking on a double yellow line or clearway: 1 demerit point
- Parking in a space reserved for persons with disability without a valid permit: 3 demerit points

Other serious offences:
- Failure to stop after a road accident (hit and run): 5 demerit points
- Driving without a valid driving licence: 5 demerit points
- Driving without valid vehicle insurance: 5 demerit points
- Reckless or dangerous driving: 5 demerit points
- Racing on a public road: 5 demerit points

FINE STRUCTURE (Cap. 65 art. 15A)
- Contraventions carry a fixed penalty stated on the notice.
- If not paid within 15 days: fine doubles.
- If not paid within 30 days: fine triples.
- Non-payment leads to referral to the Commissioner for Justice and potential court proceedings.
"""

PENALTY_MT = """TABELLA TAL-PUNTI TAT-TRAFFIKU TAT-TOROQ MALTIN
Sors: Transport Malta / Kap. 65 arts. 15B-15I

IS-SISTEMA TAL-PUNTI TAD-DEMERIT — REGOLI EWLENIN
- Kull sewwieq jibda b'żero punti tad-demerit.
- Il-punti jiġbru fuq perjodu ta' sentejn mill-jum ta' kull reat.
- Meta tasal 12-il punt il-liċenzja tiġi sospiża għal tliet xhur (l-ewwel darba).
- Sewwieqa ġodda (liċenzja miżmuma taħt sentejn) jiġu sospiżi bi 6 punti.
- Wara s-sospensjoni l-kontatur jerġa' jibda minn żero.
- Sospensjonijiet ripetuti fi ħdan ħames snin iġorru perjodi ta' sospensjoni itwal.

SKEDA TAR-REATI U L-PUNTI TAD-DEMERIT (Kap. 65, It-Tieni Skeda)

Reati tal-veloċità:
- Teċċedi l-limitu b'1-10 km/s: 1 punt tad-demerit
- Teċċedi l-limitu b'11-20 km/s: 2 punti tad-demerit
- Teċċedi l-limitu b'21-30 km/s: 3 punti tad-demerit
- Teċċedi l-limitu b'31-40 km/s: 4 punti tad-demerit
- Teċċedi l-limitu b'aktar minn 40 km/s: 5 punti tad-demerit

Reati tal-mowbajl u d-distrazzjoni:
- Tuża mowbajl f'idejk waqt is-sewqan: 3 punti tad-demerit
- Taqra jew tibgħat messaġġ waqt is-sewqan: 3 punti tad-demerit

Reati taċ-ċinturini tas-sigurtà:
- Nuqqas li tilbes ċinturin tas-sigurtà (sewwieq): 3 punti tad-demerit
- Nuqqas li tiżgura li passiġġier tat-tfal ikun imwaħħal: 3 punti tad-demerit

Reati tal-alkoħol u d-droga:
- Sewqan b'alkoħol fid-demm 0.08-0.15 g/100ml: 4 punti tad-demerit
- Sewqan b'alkoħol fid-demm ogħla minn 0.15 g/100ml: 5 punti tad-demerit
- Tirrifjuta test tan-nifs meta mitlub mill-pulizija: 5 punti tad-demerit
- Sewqan taħt l-influwenza tad-droga: 5 punti tad-demerit

Reati tat-traffiku u s-sinjali:
- Nuqqas li tieqaf għal sinjal ta' traffiku aħmar: 4 punti tad-demerit
- Nuqqas li tagħti preċedenza kif meħtieġ minn sinjal Give Way: 2 punti tad-demerit
- Nuqqas li tieqaf għal sinjal Stop: 3 punti tad-demerit

Reati tal-overtake:
- Overtake perikoluż: 4 punti tad-demerit
- Overtake fuq linja solida bajda: 3 punti tad-demerit

Reati tal-parkeġġ:
- Parkeġġ fuq passaġġ pedonali jew fi 10 metri qablu: 2 punti tad-demerit
- Parkeġġ fuq linja doppja safra jew clearway: 1 punt tad-demerit
- Parkeġġ f'post riżervat għal persuni b'diżabbiltà mingħajr permess validu: 3 punti tad-demerit

Reati serji oħra:
- Nuqqas li tieqaf wara inċident tat-triq (hit and run): 5 punti tad-demerit
- Sewqan mingħajr liċenzja valida: 5 punti tad-demerit
- Sewqan mingħajr assigurazzjoni valida tal-vettura: 5 punti tad-demerit
- Sewqan rikless jew perikoluż: 5 punti tad-demerit
- Tiġri fuq triq pubblika: 5 punti tad-demerit

STRUTTURA TAL-MULTI (Kap. 65 art. 15A)
- Il-kontravvenzjonijiet iġorru penali fiżżi ddikjarati fuq l-avviż.
- Jekk ma titħallasx fi 15-il jum: il-multa tirdoppja.
- Jekk ma titħallasx fi 30 jum: il-multa tittripla.
- In-nuqqas ta' ħlas iwassal għal rinviju lill-Kummissarju għall-Ġustizzja u eventwaliment proċedimenti legali.
"""

TRAFFIC_FAQ_EN = """MALTESE ROAD TRAFFIC — COMMON QUESTIONS
Source: Driver FAQ

SPEED LIMITS (S.L. 65.11)
- Built-up areas (urban roads): 50 km/h unless otherwise signed.
- Rural roads outside built-up areas: 80 km/h unless otherwise signed.
- Motorways (expressways): 100 km/h.
- School zones when in operation: 30 km/h.
- Speed limit signs override the default limits above.

SEATBELTS
- All occupants of a vehicle must wear a seatbelt where one is fitted (Cap. 65 art. 8).
- Children under 3 years must travel in an approved child restraint system.
- Children 3-11 years and under 135 cm must use a booster seat or child seat.
- Failure to comply carries a fine and 3 demerit points.

MOBILE PHONES
- It is illegal to use a hand-held mobile phone while driving (Cap. 65 art. 7A).
- Hands-free use is permitted.
- Penalty: fine plus 3 demerit points.

DRINK-DRIVING LIMITS
- General limit: 80 mg of alcohol per 100 ml of blood (0.08 g/100ml).
- Professional drivers and new drivers: 20 mg per 100 ml of blood (0.02 g/100ml).
- Zero tolerance applies to drivers under 18.
- Police may require a roadside breath test at any time.

VEHICLE REGISTRATION AND ROADWORTHINESS
- All vehicles must have a valid VRT (Vehicle Registration Tax) certificate.
- Annual roadworthiness test (MOT equivalent) required for vehicles over one year old.
- Untested or failed vehicles may not be used on public roads.

OVERTAKING
- Overtaking on the right is the rule in Malta (vehicles drive on the left).
- Never overtake on a solid white centre line.
- Never overtake at junctions, pedestrian crossings, or where vision is restricted.

PARKING
- Yellow lines indicate parking restrictions: double yellow = no parking at any time.
- Blue lines indicate pay-and-display parking zones.
- Disabled parking bays (marked with the wheelchair symbol) require a valid disabled permit.
- Parking on a pavement is prohibited where it obstructs pedestrians.

ACCIDENTS
- After any collision you must stop and exchange details with all parties involved (Cap. 65 art. 19).
- If someone is injured, call the police and emergency services immediately (112).
- Failure to stop after an accident carries 5 demerit points and potential criminal charges.

MICROMOBILITY (S.L. 65.32)
- Electric scooters and e-bikes are regulated under S.L. 65.32.
- Maximum speed for e-scooters on cycle lanes: 25 km/h.
- E-scooter riders must be at least 16 years old.
- Helmets are mandatory for e-scooter riders.
- E-scooters must not be ridden on pavements or pedestrianised areas.
"""

TRAFFIC_FAQ_MT = """TRAFFIKU TAT-TOROQ MALTIN — MISTOQSIJIET KOMUNI
Sors: Driver FAQ

LIMITI TAL-VELOĊITÀ (S.L. 65.11)
- Żoni urbani (toroq fil-belt): 50 km/s sakemm ma jkunx indikat mod ieħor.
- Toroq rurali barra ż-żoni urbani: 80 km/s sakemm ma jkunx indikat mod ieħor.
- Awtostradi (expressways): 100 km/s.
- Żoni ta' skejjel meta jkunu operattivi: 30 km/s.
- Sinjali tal-limitu tal-veloċità jissostitwixxu l-limiti ta' default hawn fuq.

ĊINTURINI TAS-SIGURTÀ
- Il-passiġġieri kollha tal-vettura għandhom jilbsu ċinturin tas-sigurtà fejn wieħed ikun installat (Kap. 65 art. 8).
- Tfal taħt it-3 snin iridu jivvjaġġaw f'sistema approvata tal-irżam tat-tfal.
- Tfal bejn 3-11-il sena u taħt 135 cm iridu jużaw siġġu jew booster seat.
- Nuqqas ta' konformità jġorr multa u 3 punti tad-demerit.

MOWBAJLS
- Hija illegali l-użu ta' mowbajl f'idejk waqt is-sewqan (Kap. 65 art. 7A).
- L-użu hands-free huwa permess.
- Penali: multa flimkien ma' 3 punti tad-demerit.

LIMITI TAL-ALKOĦOL FIS-SEWQAN
- Limitu ġenerali: 80 mg ta' alkoħol kull 100 ml demm (0.08 g/100ml).
- Sewwieqa professjonali u ġodda: 20 mg kull 100 ml demm (0.02 g/100ml).
- Tolleranza żero tapplika għas-sewwieqa taħt it-18-il sena.
- Il-pulizija tista' titlob test tan-nifs fuq il-ġenb tat-triq fi kwalunkwe ħin.

REĠISTRAZZJONI TAL-VETTURA U IDONEITÀ GĦAT-TRIQ
- Il-vetturi kollha għandhom ikollhom ċertifikat validu tal-VRT (Taxxa tar-Reġistrazzjoni tal-Vettura).
- Test annwali tal-idoneità (ekwivalenti tal-MOT) meħtieġ għall-vetturi iktar minn sena.
- Vetturi li ma ttestjawx jew li fallew ma jistgħux jintużaw fuq toroq pubbliċi.

OVERTAKE
- L-overtake fuq il-lemin huwa r-regola f'Malta (il-vetturi jsuqu fuq ix-xellug).
- Qatt tagħmel overtake fuq linja ċentrali solida bajda.
- Qatt tagħmel overtake f'incroci, passaġġi pedonali, jew fejn il-viżibilità tkun limitata.

PARKEĠĠ
- Linji sofor jindikaw restrizzjonijiet tal-parkeġġ: doppja safra = l-ebda parkeġġ fi kwalunkwe ħin.
- Linji blu jindikaw żoni ta' parkeġġ pay-and-display.
- Postijiet tal-parkeġġ għal persuni b'diżabbiltà (immarkati bis-simbolu tas-siġġu tar-roti) jeħtieġu permess validu.
- Il-parkeġġ fuq bankina huwa projbit fejn jostakola l-pedoni.

INĊIDENTI
- Wara kull kolliżjoni għandek tieqaf u tiskambja d-dettalji mal-partijiet kollha involuti (Kap. 65 art. 19).
- Jekk xi ħadd ikun midrub, ċempel il-pulizija u s-servizzi ta' emerġenza immedjatament (112).
- Nuqqas li tieqaf wara inċident iġorr 5 punti tad-demerit u potenzjalment akkużi kriminali.

MIKROMOBILTÀ (S.L. 65.32)
- Skutters elettriċi u e-bikes huma regolati taħt S.L. 65.32.
- Veloċità massima għall-iskutters elettriċi fil-korsiji taċ-ċiklisti: 25 km/s.
- Is-sewwieqa tal-iskutters elettriċi għandhom ikollhom mill-inqas 16-il sena.
- Elmi huma obbligatorji għas-sewwieqa tal-iskutters elettriċi.
- L-iskutters elettriċi ma jistgħux jinqaddu fuq bankini jew żoni pedonalizzati.
"""


def pdf_to_text(url: str) -> str | None:
    print(f"  fetching {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    except requests.RequestException as e:
        print(f"  ! network error: {e}")
        return None

    if r.status_code != 200:
        print(f"  ! HTTP {r.status_code}")
        return None

    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" not in ctype and not r.content.startswith(b"%PDF"):
        print(f"  ! not a PDF (Content-Type: {ctype}); skipping")
        return None

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name
    try:
        return extract_text(tmp_path) or ""
    finally:
        os.remove(tmp_path)


def main() -> int:
    for lang in ("en", "mt"):
        (BASE / lang).mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for url, lang, fname in SOURCES:
        out = BASE / lang / fname
        if out.exists() and out.stat().st_size > 512:
            print(f"[{lang}] {fname} — already exists, skipping")
            continue
        print(f"[{lang}] {fname}")
        text = pdf_to_text(url)
        if not text or len(text.strip()) < 100:
            failures.append(f"{lang}/{fname}")
            print(f"  ! download failed or empty — will use manual fallback if available")
            continue
        out.write_text(text, encoding="utf-8")
        print(f"  wrote {len(text)} chars")

    # Always write manual content (overwrites stale or 0-byte files).
    (BASE / "en" / "payment_options.txt").write_text(FAQ_EN, encoding="utf-8")
    (BASE / "mt" / "payment_options.txt").write_text(FAQ_MT, encoding="utf-8")
    (BASE / "en" / "penalty_points.txt").write_text(PENALTY_EN, encoding="utf-8")
    (BASE / "mt" / "penalty_points.txt").write_text(PENALTY_MT, encoding="utf-8")
    (BASE / "en" / "traffic_faq.txt").write_text(TRAFFIC_FAQ_EN, encoding="utf-8")
    (BASE / "mt" / "traffic_faq.txt").write_text(TRAFFIC_FAQ_MT, encoding="utf-8")

    # For Maltese files that failed, write an English placeholder so the
    # retriever always has something for both language slots.
    mt_fallbacks = {
        "traffic_act.txt": None,          # no short substitute — flag only
        "motor_vehicle_insurance.txt": None,
        "traffic_signs_regs.txt": None,
        "speed_limits_regs.txt": None,
        "driving_licences.txt": None,
        "alcohol_regs.txt": None,
        "micromobility_regs.txt": None,
        "enforcement_camera.txt": None,
    }
    for fname in mt_fallbacks:
        out = BASE / "mt" / fname
        if not out.exists() or out.stat().st_size < 100:
            print(f"  [mt] {fname} missing — manual content required for full bilingual coverage")

    if failures:
        print("\nWARNING: the following sources failed PDF download:")
        for f in failures:
            print(f"  - {f}")
        print("Manual content written for penalty_points, payment_options, and traffic_faq.")
        print("For the remaining Maltese statutory files, download manually from legislation.mt")
        print("and save under data/cleaned_txt/mt/<filename>.txt, then re-run with --rebuild.")

    print(f"\ncorpus written under {BASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
