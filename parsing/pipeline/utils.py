from __future__ import annotations

import re
from typing import Optional


_SPACE_RE = re.compile(r"[\s\u00A0]+")  # обычные + неразрывные пробелы


def normalize_spaces(s: str) -> str:
    return _SPACE_RE.sub(" ", s).strip()


def parse_salary_rub(s: object) -> Optional[float]:
    """
    Примеры:
    "27 000 руб." -> 27000
    "150000" -> 150000
    "" / NaN -> None
    """
    if s is None:
        return None
    text = str(s)
    text = normalize_spaces(text).lower()
    if text in ("", "nan", "none"):
        return None

    # оставляем только цифры
    digits = re.findall(r"\d+", text)
    if not digits:
        return None
    return float("".join(digits))


def parse_gender_age(s: object) -> tuple[Optional[int], Optional[int]]:
    """
    Из поля вида: "Мужчина , 42 года , родился ..."
    gender: 1=мужчина, 0=женщина, None=неизвестно
    age: целое число, None=не распарсили
    """
    if s is None:
        return None, None
    text = normalize_spaces(str(s)).lower()

    gender = None
    if "муж" in text:
        gender = 1
    elif "жен" in text:
        gender = 0

    m = re.search(r"(\d{1,3})\s*год", text)
    age = int(m.group(1)) if m else None

    return gender, age


def parse_experience_months(s: object) -> Optional[int]:
    """
    В HH встречается кусок: "Опыт работы 6 лет 1 месяц"
    Иногда просто длинный текст, но там тоже есть "6 лет 1 месяц".
    Переводим в месяцы.
    """
    if s is None:
        return None
    text = normalize_spaces(str(s)).lower()

    # ищем "X лет" и "Y месяцев"
    years = 0
    months = 0

    my = re.search(r"(\d+)\s*лет|\b(\d+)\s*год", text)
    m_years = re.search(r"(\d+)\s*(лет|года|год)", text)
    if m_years:
        years = int(m_years.group(1))

    m_months = re.search(r"(\d+)\s*(месяц|месяца|месяцев)", text)
    if m_months:
        months = int(m_months.group(1))

    total = years * 12 + months
    return total if total > 0 else None


def parse_has_car(s: object) -> int:
    """
    Поле "Авто"
    """
    if s is None:
        return 0
    text = normalize_spaces(str(s)).lower()
    if text in ("", "nan", "none"):
        return 0
    # любые позитивные признаки
    if "имеется" in text or "собствен" in text or "автомоб" in text:
        return 1
    return 0


def clean_city(s: object) -> str:
    """
    "Липецк , не готов к переезду , ..." -> "Липецк"
    """
    if s is None:
        return ""
    text = normalize_spaces(str(s))
    if not text:
        return ""
    return text.split(",")[0].strip()
