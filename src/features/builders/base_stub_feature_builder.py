from __future__ import annotations

import pandas as pd

from src.features.contracts.feature_builder_contract import FeatureBuilderContract
from src.features.models.feature_context import FeatureContext


class BaseStubFeatureBuilder(FeatureBuilderContract):
    block_name = "stub"

    def provides(self) -> set[str]:
        return set()

    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        return self._build_stub(context)

    def _build_stub(self, context: FeatureContext) -> pd.DataFrame:
        return pd.DataFrame(index=context.main_frame.index)
