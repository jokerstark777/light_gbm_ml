from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.features.models.feature_context import FeatureContext


class FeatureBuilderContract(ABC):
    block_name: str

    @abstractmethod
    def provides(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        raise NotImplementedError
