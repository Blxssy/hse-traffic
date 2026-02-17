from __future__ import annotations
import re
from typing import Optional

_SPACE_RE = re.compile(r"[\s\u00A0]+")


def norm_spaces(s: str) -> str:
    return _SPACE_RE.sub(" ", s).strip()


def parse_salary_rub(val: object) -> Optional[float]:
    if val is None:
        return None
    txt = norm_spaces(str(val)).lower()
    if txt in ("", "nan", "none"):
        return None
    digits = re.findall(r"\d+", txt)
    if not digits:
        return None
    return float("".join(digits))


def parse_gender_age(val: object) -> tuple[Optional[int], Optional[int]]:
    if val is None:
        return None, None
    txt = norm_spaces(str(val)).lower()
    gender = None
    if "муж" in txt:
        gender = 1
    elif "жен" in txt:
        gender = 0
    m = re.search(r"(\d{1,3})\s*год", txt)
    age = int(m.group(1)) if m else None
    return gender, age


def parse_experience_months(val: object) -> Optional[int]:
    if val is None:
        return None
    txt = norm_spaces(str(val)).lower()

    y = 0
    mth = 0
    my = re.search(r"(\d+)\s*(лет|года|год)", txt)
    if my:
        y = int(my.group(1))
    mm = re.search(r"(\d+)\s*(месяц|месяца|месяцев)", txt)
    if mm:
        mth = int(mm.group(1))

    total = y * 12 + mth
    return total if total > 0 else None


def clean_city(val: object) -> str:
    if val is None:
        return ""
    txt = norm_spaces(str(val))
    return txt.split(",")[0].strip() if txt else ""
