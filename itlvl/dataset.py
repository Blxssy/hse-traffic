from __future__ import annotations

from pathlib import Path
import pandas as pd

from itlvl.config import AppConfig
from itlvl.utils import parse_salary_rub, parse_gender_age, parse_experience_months, clean_city
from itlvl.labeling import infer_level


IT_TITLE_KEYWORDS = (
    "разработчик", "developer", "dev", "programmer", "software", "backend", "frontend",
    "fullstack", "golang", "go ", "java", "python", "javascript", "typescript", "node",
    "c#", "cpp", "c++", "php", "kotlin", "swift", "ios", "android", "django", "flask",
    "spring", "react", "vue", "angular", ".net"
)


def load_hh_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        encoding="utf-8",
        on_bad_lines="skip",
    )


def is_it_resume(title: str, blob: str) -> bool:
    t = (title or "").lower()
    b = (blob or "").lower()
    return any(k in t for k in IT_TITLE_KEYWORDS) or any(k in b for k in IT_TITLE_KEYWORDS)


def make_it_dataset(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    title_col = "Ищет работу на должность:"
    ga_col = "Пол, возраст"
    sal_col = "ЗП"
    city_col = "Город"
    exp_col = "Опыт (двойное нажатие для полной версии)"
    skills_col = "Ключевые навыки" if "Ключевые навыки" in df.columns else None

    for c in [title_col, ga_col, sal_col, city_col, exp_col]:
        if c not in df.columns:
            raise KeyError(f"Column not found: {c}")

    out = df.copy()

    out["title"] = out[title_col].astype(str).fillna("")
    ga = out[ga_col].map(parse_gender_age)
    out["gender"] = ga.map(lambda t: t[0])
    out["age"] = ga.map(lambda t: t[1])
    out["salary_rub"] = out[sal_col].map(parse_salary_rub)
    out["city"] = out[city_col].map(clean_city)
    out["exp_months"] = out[exp_col].map(parse_experience_months)

    if skills_col:
        out["skills"] = out[skills_col].astype(str).fillna("")
    else:
        out["skills"] = ""

    out["blob"] = out["title"].astype(str) + " " + out["skills"].astype(str) + " " + out[exp_col].astype(str)
    out = out[out.apply(lambda r: is_it_resume(r["title"], r["blob"]), axis=1)].copy()

    out = out.dropna(subset=["age", "gender", "exp_months", "salary_rub"]).copy()
    out = out[(out["salary_rub"] >= 10_000) & (out["salary_rub"] <= 1_000_000)].copy()
    out["exp_months"] = out["exp_months"].clip(0, 720)

    out["level"] = out.apply(lambda r: infer_level(r["title"], int(r["exp_months"]), cfg), axis=1)

    out["title"] = out["title"].fillna("").astype(str)
    out["skills"] = out["skills"].fillna("").astype(str)

    return out[["salary_rub", "city", "age", "gender", "exp_months", "title", "skills", "level"]].reset_index(drop=True)
