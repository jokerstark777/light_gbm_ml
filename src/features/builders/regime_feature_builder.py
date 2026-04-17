from __future__ import annotations

from src.features.builders.base_stub_feature_builder import BaseStubFeatureBuilder


class RegimeFeatureBuilder(BaseStubFeatureBuilder):
    def __init__(
        self,
        timeframe_label: str = "1h",
        use_htf: bool = False,
        reference_timeframe: str | None = None,
    ) -> None:
        self.timeframe_label = timeframe_label
        self.use_htf = use_htf
        self.reference_timeframe = reference_timeframe or timeframe_label
        self.block_name = f"regime_{timeframe_label}"
