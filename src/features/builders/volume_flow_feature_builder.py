from __future__ import annotations

from src.features.builders.base_stub_feature_builder import BaseStubFeatureBuilder


class VolumeFlowFeatureBuilder(BaseStubFeatureBuilder):
    block_name = "volume_flow"

    def __init__(self, timeframe_label: str = "1h", reference_timeframe: str | None = None) -> None:
        self.timeframe_label = timeframe_label
        self.reference_timeframe = reference_timeframe or timeframe_label
