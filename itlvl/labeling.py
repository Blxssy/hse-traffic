from __future__ import annotations
import re
from typing import Optional

from itlvl.config import AppConfig

JUN_PAT = re.compile(r"\b(junior|джуниор|стаж(е|ё)р|intern)\b", re.IGNORECASE)
SEN_PAT = re.compile(r"\b(senior|сеньор|lead|team\s*lead|тимлид|ведущ(ий|ая)|главн(ый|ая))\b", re.IGNORECASE)


def infer_level(title: str, exp_months: Optional[int], cfg: AppConfig) -> str:
    t = title or ""
    if JUN_PAT.search(t):
        return "junior"
    if SEN_PAT.search(t):
        return "senior"

    if exp_months is None:
        return "middle"
    if exp_months <= cfg.junior_max_months:
        return "junior"
    if exp_months >= cfg.senior_min_months:
        return "senior"
    return "middle"
