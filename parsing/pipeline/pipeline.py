from __future__ import annotations

from parsing.pipeline.handlers.load_csv import LoadCSVHandler
from parsing.pipeline.handlers.clean_columns import CleanColumnsHandler
from parsing.pipeline.handlers.parse_target_salary import ParseTargetSalaryHandler
from parsing.pipeline.handlers.parse_basic_fields import ParseBasicFieldsHandler
from parsing.pipeline.handlers.build_features import BuildFeaturesHandler
from parsing.pipeline.handlers.save_npy import SaveNpyHandler
from parsing.pipeline.handlers.base import Handler


def build_pipeline() -> Handler:
    """
    Chain of Responsibility:
      LoadCSV -> CleanColumns -> ParseSalary(y) -> ParseBasic -> BuildFeatures -> SaveNpy
    """
    h1 = LoadCSVHandler()
    h2 = CleanColumnsHandler()
    h3 = ParseTargetSalaryHandler()
    h4 = ParseBasicFieldsHandler()
    h5 = BuildFeaturesHandler()
    h6 = SaveNpyHandler()

    h1.set_next(h2).set_next(h3).set_next(h4).set_next(h5).set_next(h6)
    return h1
