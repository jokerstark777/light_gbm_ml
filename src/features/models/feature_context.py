from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureContext:
    symbol: str
    main_frame: pd.DataFrame
    htf_frame: pd.DataFrame | None = None
    market_frame_map: dict[str, pd.DataFrame] | None = None
