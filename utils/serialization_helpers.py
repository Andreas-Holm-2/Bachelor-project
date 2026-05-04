import asyncio
import os
import pickle

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


def list_to_string(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, np.ndarray)):
        items = [str(v) for v in val if v and not (isinstance(v, float) and pd.isna(v))]
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f" and {items[-1]}"
    s = str(val).strip()
    return s if s else None


def parse_year(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


QUALIFIED_COLS = {
    "position_held", "educated_at", "award", "participant_in",
    "member_of_sports_team", "employer", "spouse", "student_of",
}


def format_qualified_entry(entry):
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    if not value:
        return None

    start  = entry.get("start")
    end    = entry.get("end")
    pit    = entry.get("point_in_time")
    degree = entry.get("degree")

    start_year = start[:4] if isinstance(start, str) else None
    end_year   = end[:4]   if isinstance(end, str)   else None
    pit_year   = pit[:4]   if isinstance(pit, str)   else None
    degree_str = f" ({degree})" if degree else ""

    if start_year and end_year:
        return f"{value}{degree_str} from {start_year} to {end_year}"
    elif start_year:
        return f"{value}{degree_str} from {start_year}"
    elif end_year:
        return f"{value}{degree_str} until {end_year}"
    elif pit_year:
        return f"{value}{degree_str} in {pit_year}"
    return f"{value}{degree_str}" if degree_str else value


def qualified_to_string(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        items = [format_qualified_entry(v) for v in val if isinstance(v, dict)]
        items = [i for i in items if i]
        return list_to_string(items)
    return list_to_string(val)


def get_field_value(row, col):
    if col in ("birth_year", "death_year"):
        val = parse_year(row.get(col))
        return str(val) if val is not None else None
    if col in ("name", "gender"):
        return str(row.get(col)) if pd.notna(row.get(col)) else None
    if col in QUALIFIED_COLS:
        return qualified_to_string(row.get(col))
    return list_to_string(row.get(col))


# All columns present in the shared dataframe, in display order.
# Pass exclude_cols to build_json / build_template to omit specific columns.
FIELD_LABELS = [
    ("name",                  "Name"),
    ("gender",                "Gender"),
    ("birth_year",            "Birth year"),
    ("death_year",            "Death year"),
    ("birth_place",           "Birth place"),
    ("death_place",           "Death place"),
    ("burial_place",          "Burial place"),
    ("birth_name",            "Birth name"),
    ("height",                "Height"),
    ("manner_of_death",       "Manner of death"),
    ("occupation",            "Occupation"),
    ("position_held",         "Position held"),
    ("position_played",       "Position played"),
    ("field_of_work",         "Field of work"),
    ("genre",                 "Genre"),
    ("notable_work",          "Notable work"),        # added
    ("instrument",            "Instrument"),
    ("employer",              "Employer"),
    ("educated_at",           "Educated at"),
    ("student_of",            "Student of"),
    ("student",               "Students"),            # added
    ("academic_degree",       "Academic degree"),
    ("military_rank",         "Military rank"),
    ("political_party",       "Political party"),
    ("affiliation",           "Affiliation"),
    ("member_of",             "Member of"),
    ("member_of_sports_team", "Member of sports team"),
    ("award",                 "Award"),
    ("language",              "Languages"),
    ("native_language",       "Native language"),
    ("religion",              "Religion"),
    ("sport",                 "Sport"),
    ("sports_discipline",     "Sports discipline"),   # added
    ("work_location",         "Work location"),
    ("residence",             "Residence"),
    ("participant_in",        "Participant in"),
    ("spouse",                "Spouse"),
    ("child",                 "Children"),
    ("sibling",               "Siblings"),
    ("relative",              "Relatives"),
    ("family",                "Family"),
    ("father",                "Father"),
    ("mother",                "Mother"),
]


def build_json(row, *, exclude_cols=frozenset()):
    """Build a key-value text serialization for a person row.

    Parameters
    ----------
    exclude_cols : set
        Columns to omit. Use ``{"birth_year"}`` for birth-year non-trivial,
        ``{"name", "gender"}`` for gender non-trivial, etc.
    """
    parts = []
    for col, label in FIELD_LABELS:
        if col in exclude_cols:
            continue
        val = get_field_value(row, col)
        if val:
            parts.append(f"{label}: {val}")
    return ". ".join(parts) + "." if parts else ""


def build_template(row, *, exclude_cols=frozenset(), use_name=True, use_gender_pronouns=True):
    """Build a biographical F-string sentence for a person row.

    Parameters
    ----------
    exclude_cols : set
        Columns whose information should not appear in the output.
        Use ``{"birth_year"}`` for birth-year non-trivial tasks, and
        ``{"gender", "spouse", "father", "mother", "child"}`` for gender
        non-trivial tasks (prevents indirect gender leakage via names).
    use_name : bool
        If False, replace the person's name with "This person" — used for
        gender non-trivial to avoid name-based gender signals.
    use_gender_pronouns : bool
        If False, use neutral "They/Their" regardless of gender — used for
        gender non-trivial.
    """
    gender = str(row.get("gender")) if pd.notna(row.get("gender")) else None

    show_name   = use_name   and "name"   not in exclude_cols
    show_gender = use_gender_pronouns and "gender" not in exclude_cols

    name = (str(row.get("name")) if pd.notna(row.get("name")) else "This person") if show_name else "This person"

    if show_gender and gender:
        pronoun     = {"male": "He",  "female": "She" }.get(gender, "They")
        pronoun_pos = {"male": "His", "female": "Her" }.get(gender, "Their")
    else:
        pronoun     = "They"
        pronoun_pos = "Their"

    birth_year  = parse_year(row.get("birth_year")) if "birth_year" not in exclude_cols else None
    death_year  = parse_year(row.get("death_year"))
    birth_place = list_to_string(row.get("birth_place"))
    death_place = list_to_string(row.get("death_place"))

    parts = []

    # Opening: "Name was born in 1902 and died in 1955 and identified as male."
    year_clause   = ""
    if birth_year and death_year:
        year_clause = f"was born in {birth_year} and died in {death_year}"
    elif birth_year:
        year_clause = f"was born in {birth_year}"
    elif death_year:
        year_clause = f"died in {death_year}"

    gender_clause = f"identified as {gender}" if "gender" not in exclude_cols and gender else ""

    opening_clauses = [c for c in [year_clause, gender_clause] if c]
    if opening_clauses:
        parts.append(f"{name} " + " and ".join(opening_clauses) + ".")
    else:
        parts.append(f"{name}.")

    if birth_place and death_place:
        parts.append(f"{pronoun} was born in {birth_place} and died in {death_place}.")
    elif birth_place:
        parts.append(f"{pronoun} was born in {birth_place}.")
    elif death_place:
        parts.append(f"{pronoun} died in {death_place}.")

    burial = list_to_string(row.get("burial_place"))
    if burial and "burial_place" not in exclude_cols:
        parts.append(f"{pronoun} was buried in {burial}.")

    bn = list_to_string(row.get("birth_name"))
    if bn and "birth_name" not in exclude_cols:
        parts.append(f"{pronoun_pos} birth name was {bn}.")

    height = list_to_string(row.get("height"))
    if height and "height" not in exclude_cols:
        parts.append(f"{pronoun_pos} height was {height}.")

    mod = list_to_string(row.get("manner_of_death"))
    if mod and "manner_of_death" not in exclude_cols:
        parts.append(f"{pronoun_pos} manner of death was {mod}.")

    occ = list_to_string(row.get("occupation"))
    if occ and "occupation" not in exclude_cols:
        parts.append(f"{pronoun} worked as {occ}.")

    pos = qualified_to_string(row.get("position_held"))
    if pos and "position_held" not in exclude_cols:
        parts.append(f"{pronoun} held the position of {pos}.")

    pp = list_to_string(row.get("position_played"))
    if pp and "position_played" not in exclude_cols:
        parts.append(f"{pronoun} played the position of {pp}.")

    fow = list_to_string(row.get("field_of_work"))
    if fow and "field_of_work" not in exclude_cols:
        parts.append(f"{pronoun_pos} field of work was {fow}.")

    notable_work = list_to_string(row.get("notable_work"))
    if notable_work and "notable_work" not in exclude_cols:
        parts.append(f"{pronoun_pos} notable works included {notable_work}.")

    genre = list_to_string(row.get("genre"))
    if genre and "genre" not in exclude_cols:
        parts.append(f"{pronoun_pos} genre was {genre}.")

    instr = list_to_string(row.get("instrument"))
    if instr and "instrument" not in exclude_cols:
        parts.append(f"{pronoun} played {instr}.")

    emp = qualified_to_string(row.get("employer"))
    if emp and "employer" not in exclude_cols:
        parts.append(f"{pronoun} was employed by {emp}.")

    edu = qualified_to_string(row.get("educated_at"))
    if edu and "educated_at" not in exclude_cols:
        parts.append(f"{pronoun} studied at {edu}.")

    stu = qualified_to_string(row.get("student_of"))
    if stu and "student_of" not in exclude_cols:
        parts.append(f"{pronoun} studied under {stu}.")

    student = qualified_to_string(row.get("student"))
    if student and "student" not in exclude_cols:
        parts.append(f"{pronoun_pos} students included {student}.")

    deg = list_to_string(row.get("academic_degree"))
    if deg and "academic_degree" not in exclude_cols:
        parts.append(f"{pronoun} held an academic degree in {deg}.")

    rank = list_to_string(row.get("military_rank"))
    if rank and "military_rank" not in exclude_cols:
        parts.append(f"{pronoun} held the military rank of {rank}.")

    party = list_to_string(row.get("political_party"))
    if party and "political_party" not in exclude_cols:
        parts.append(f"{pronoun} was a member of {party}.")

    aff = list_to_string(row.get("affiliation"))
    if aff and "affiliation" not in exclude_cols:
        parts.append(f"{pronoun} was affiliated with {aff}.")

    sport = list_to_string(row.get("sport"))
    if sport and "sport" not in exclude_cols:
        parts.append(f"{pronoun} played {sport}.")

    sports_discipline = list_to_string(row.get("sports_discipline"))
    if sports_discipline and "sports_discipline" not in exclude_cols:
        parts.append(f"{pronoun_pos} sports discipline was {sports_discipline}.")

    team = qualified_to_string(row.get("member_of_sports_team"))
    if team and "member_of_sports_team" not in exclude_cols:
        parts.append(f"{pronoun} was a member of sports team {team}.")

    mem = list_to_string(row.get("member_of"))
    if mem and "member_of" not in exclude_cols:
        parts.append(f"{pronoun} was a member of {mem}.")

    part = qualified_to_string(row.get("participant_in"))
    if part and "participant_in" not in exclude_cols:
        parts.append(f"{pronoun} participated in {part}.")

    aw = qualified_to_string(row.get("award"))
    if aw and "award" not in exclude_cols:
        parts.append(f"{pronoun} received {aw}.")

    lang = list_to_string(row.get("language"))
    if lang and "language" not in exclude_cols:
        parts.append(f"{pronoun} spoke {lang}.")

    nlang = list_to_string(row.get("native_language"))
    if nlang and "native_language" not in exclude_cols:
        parts.append(f"{pronoun_pos} native language was {nlang}.")

    rel = list_to_string(row.get("religion"))
    if rel and "religion" not in exclude_cols:
        parts.append(f"{pronoun_pos} religion was {rel}.")

    wl = list_to_string(row.get("work_location"))
    if wl and "work_location" not in exclude_cols:
        parts.append(f"{pronoun} worked in {wl}.")

    res = list_to_string(row.get("residence"))
    if res and "residence" not in exclude_cols:
        parts.append(f"{pronoun} resided in {res}.")

    spouse = qualified_to_string(row.get("spouse"))
    if spouse and "spouse" not in exclude_cols:
        parts.append(f"{pronoun} was married to {spouse}.")

    father = list_to_string(row.get("father")) if "father" not in exclude_cols else None
    mother = list_to_string(row.get("mother")) if "mother" not in exclude_cols else None
    if father and mother:
        parts.append(f"{pronoun} was the child of {father} and {mother}.")
    elif father:
        parts.append(f"{pronoun} was the child of {father}.")
    elif mother:
        parts.append(f"{pronoun} was the child of {mother}.")

    child = list_to_string(row.get("child"))
    if child and "child" not in exclude_cols:
        parts.append(f"{pronoun} had children including {child}.")

    sib = list_to_string(row.get("sibling"))
    if sib and "sibling" not in exclude_cols:
        parts.append(f"{pronoun_pos} siblings included {sib}.")

    relative = list_to_string(row.get("relative"))
    if relative and "relative" not in exclude_cols:
        parts.append(f"{pronoun_pos} relatives included {relative}.")

    fam = list_to_string(row.get("family"))
    if fam and "family" not in exclude_cols:
        parts.append(f"{pronoun} belonged to the {fam} family.")

    return " ".join(parts)


_LLM_SYSTEM_PROMPT = """You are a biographical description generator. You will receive facts about a person in key-value format, and your task is to:
- Convert their facts into a fluent, natural-sounding biographical paragraph (no newlines)
- Use correct pronouns (he/him/she/hey) based on the Gender field. (if none, use them/this person instead) .
- Do not make up any information; generate only the sentences based on the provided information.
- Keep the biography under a maximum of 340 words.
- Return ONLY the biography text, nothing else"""


async def generate_biographies(
    df,
    *,
    exclude_cols=frozenset(),
    model="Gemma 4",
    checkpoint_path="biographies_checkpoint.pkl",
    concurrency=8,
    save_every=500,
):
    """Generate LLM biographies for every row in *df*.

    Parameters
    ----------
    exclude_cols : set
        Columns to omit from the key-value prompt, e.g.
        ``{"name", "gender", "spouse", "father", "mother", "child", "sibling"}``
        for the non-trivial case.
    checkpoint_path : str
        Path to the pickle checkpoint file.  Existing progress is loaded
        automatically so interrupted runs can be resumed.

    Returns
    -------
    list
        One biography string per row (``None`` for any row that failed after
        all retries).
    """
    import openai  # imported here so the module loads without openai installed

    load_dotenv()
    client = openai.AsyncOpenAI(
        api_key=os.getenv("CAMPUSAI_API_KEY2"),
        base_url="https://api.campusai.compute.dtu.dk/v1",
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def _call_api(row, retries=5):
        prompt = build_json(row, exclude_cols=exclude_cols)
        async with semaphore:
            for attempt in range(retries):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    )
                    return response.choices[0].message.content.strip()
                except openai.RateLimitError:
                    await asyncio.sleep(2 ** attempt)
                except Exception:
                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    try:
        with open(checkpoint_path, "rb") as f:
            results_dict = pickle.load(f)
        print(f"Resuming from checkpoint: {len(results_dict)} already done")
    except FileNotFoundError:
        results_dict = {}

    todo = [(idx, row) for idx, row in df.iterrows() if idx not in results_dict]
    completed_since_save = 0

    pbar = tqdm(
        total=len(df), initial=len(results_dict),
        unit="persons", desc="Generating biographies",
    )

    async def _process(idx, row):
        nonlocal completed_since_save
        results_dict[idx] = await _call_api(row)
        completed_since_save += 1
        pbar.update(1)
        if completed_since_save >= save_every:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results_dict, f)
            completed_since_save = 0

    await asyncio.gather(*[_process(idx, row) for idx, row in todo])
    pbar.close()

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results_dict, f)

    return [results_dict[idx] for idx, _ in df.iterrows()]
