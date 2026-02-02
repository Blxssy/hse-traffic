from __future__ import annotations

from pipeline.handlers.load_csv import LoadCSVHandler
from pipeline.handlers.clean_columns import CleanColumnsHandler
from pipeline.handlers.parse_target_salary import ParseTargetSalaryHandler
from pipeline.handlers.parse_basic_fields import ParseBasicFieldsHandler
from pipeline.handlers.filter import FilterRowsHandler
from pipeline.handlers.build_features import BuildFeaturesHandler
from pipeline.handlers.save_npy import SaveNpyHandler
from pipeline.handlers.base import Handler


def build_pipeline() -> Handler:
    """
    Chain of Responsibility:
      LoadCSV -> CleanColumns -> ParseSalary(y) -> ParseBasic -> BuildFeatures -> SaveNpy
    """
    h1 = LoadCSVHandler()
    h2 = CleanColumnsHandler()
    h3 = ParseTargetSalaryHandler()
    h4 = ParseBasicFieldsHandler()
    h4b = FilterRowsHandler(min_salary=10_000, max_salary=1_000_000, max_exp_months=720)
    h5 = BuildFeaturesHandler()
    h6 = SaveNpyHandler()

    h1.set_next(h2).set_next(h3).set_next(h4).set_next(h4b).set_next(h5).set_next(h6)
    return h1
