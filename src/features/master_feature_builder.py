from __future__ import annotations

import pandas as pd

import config as cfg
from src.features.builders.compression_expansion_feature_builder import CompressionExpansionFeatureBuilder
from src.features.builders.entry_location_feature_builder import EntryLocationFeatureBuilder
from src.features.builders.htf_context_feature_builder import HtfContextFeatureBuilder
from src.features.builders.interaction_feature_builder import InteractionFeatureBuilder
from src.features.builders.market_context_feature_builder import MarketContextFeatureBuilder
from src.features.builders.ohlcv_baseline_feature_builder import OhlcvBaselineFeatureBuilder
from src.features.builders.regime_feature_builder import RegimeFeatureBuilder
from src.features.builders.session_context_feature_builder import SessionContextFeatureBuilder
from src.features.builders.trend_feature_builder import TrendFeatureBuilder
from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext
from src.features.models.feature_pipeline_result import FeaturePipelineResult
from src.features.models.feature_request import ResolvedFeatureRequest, resolve_feature_request


class MasterFeatureBuilder:
    def __init__(self) -> None:
        main_timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
        htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        main_reference_timeframe = str(getattr(cfg, "WINDOW_REFERENCE_TIMEFRAME", "5m"))
        htf_reference_timeframe = str(getattr(cfg, "HTF_WINDOW_REFERENCE_TIMEFRAME", htf_timeframe))
        self.main_builders: list[FeatureBuilderContract] = [
            OhlcvBaselineFeatureBuilder(),
            TrendFeatureBuilder(timeframe_label=main_timeframe, reference_timeframe=main_reference_timeframe),
            RegimeFeatureBuilder(timeframe_label=main_timeframe, reference_timeframe=main_reference_timeframe),
            CompressionExpansionFeatureBuilder(
                timeframe_label=main_timeframe,
                reference_timeframe=main_reference_timeframe,
            ),
            EntryLocationFeatureBuilder(
                timeframe_label=main_timeframe,
                reference_timeframe=main_reference_timeframe,
            ),
            InteractionFeatureBuilder(),
            MarketContextFeatureBuilder(),
            SessionContextFeatureBuilder(),
        ]
        self.htf_builders: list[FeatureBuilderContract] = [
            HtfContextFeatureBuilder(),
            TrendFeatureBuilder(
                timeframe_label=htf_timeframe,
                use_htf=True,
                reference_timeframe=htf_reference_timeframe,
            ),
            RegimeFeatureBuilder(
                timeframe_label=htf_timeframe,
                use_htf=True,
                reference_timeframe=htf_reference_timeframe,
            ),
        ]

    def build(
        self,
        base_candle_map: dict[str, pd.DataFrame],
        htf_candle_map: dict[str, pd.DataFrame] | None = None,
    ) -> FeaturePipelineResult:
        request = self._resolve_request()
        feature_map: dict[str, pd.DataFrame] = {}

        for symbol, main_frame in base_candle_map.items():
            base_frame = main_frame.copy().reset_index(drop=True)
            base_frame["timestamp"] = pd.to_datetime(base_frame["timestamp"], errors="coerce")
            context = FeatureContext(
                symbol=symbol,
                main_frame=base_frame,
                htf_frame=(htf_candle_map or {}).get(symbol),
                market_frame_map=base_candle_map,
            )

            built_frames = []
            for builder in [*self.main_builders, *self.htf_builders]:
                block_request = builder.provides().intersection(request.active_features)
                if not block_request:
                    continue
                built = builder.build(context, set(request.active_features))
                if built is not None and not built.empty:
                    built_frames.append(built.reset_index(drop=True))

            merged = base_frame.copy()
            for built in built_frames:
                for column in built.columns:
                    if column in merged.columns:
                        continue
                    merged[column] = built[column]
            feature_map[symbol] = merged

        return FeaturePipelineResult(
            profile_name=request.profile,
            active_blocks=request.active_blocks,
            feature_columns=request.active_features,
            feature_map=feature_map,
        )

    def _resolve_request(self) -> ResolvedFeatureRequest:
        block_features = self._collect_block_features()
        raw_request = getattr(cfg, "FEATURE_BUILD_REQUEST", {})
        profile_map = getattr(cfg, "FEATURE_PROFILES", {})
        return resolve_feature_request(raw_request, profile_map, block_features)

    def _collect_block_features(self) -> dict[str, set[str]]:
        block_features: dict[str, set[str]] = {}
        for builder in [*self.main_builders, *self.htf_builders]:
            block_features.setdefault(builder.block_name, set()).update(builder.provides())
        return block_features
