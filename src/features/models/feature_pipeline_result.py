from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeaturePipelineResult:
    profile_name: str
    active_blocks: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_map: dict[str, pd.DataFrame]
