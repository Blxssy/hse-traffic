from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler
from pipeline.utils import (
    parse_gender_age,
    parse_experience_months,
    parse_has_car,
    clean_city,
)


class ParseBasicFieldsHandler(Handler):
    def _process(self, ctx: PipelineContext) -> None:
        if ctx.df is None:
            raise RuntimeError("df is not loaded")

        df = ctx.df

        gender_age_col = "Пол, возраст"
        exp_col = "Опыт (двойное нажатие для полной версии)"
        city_col = "Город"
        employment_col = "Занятость"
        schedule_col = "График"
        car_col = "Авто"

        for col in [gender_age_col, exp_col, city_col, employment_col, schedule_col, car_col]:
            if col not in df.columns:
                raise KeyError(f"Column not found: {col}")

        ga = df[gender_age_col].map(parse_gender_age)
        df["_gender"] = ga.map(lambda t: t[0])
        df["_age"] = ga.map(lambda t: t[1])

        df["_exp_months"] = df[exp_col].map(parse_experience_months)
        df["_city"] = df[city_col].map(clean_city)
        df["_employment"] = df[employment_col].astype(str).fillna("")
        df["_schedule"] = df[schedule_col].astype(str).fillna("")
        df["_has_car"] = df[car_col].map(parse_has_car).astype("int32")

        df = df.dropna(subset=["_age", "_gender", "_exp_months"]).copy()

        ctx.df = df
