"""General types."""

from collections.abc import Callable, Sequence

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing_extensions import TypeAlias


ClassificationReport: TypeAlias = dict[str, dict[str, float] | float]
ClassWeights: TypeAlias = dict[int, float]
Tokens: TypeAlias = list[str]
FeatureExtractor: TypeAlias = BaseEstimator
FeatureExtractorMethod: TypeAlias = Callable[[Tokens], int]
FeatureExtractorMethodData: TypeAlias = (
  list[tuple[str, FeatureExtractorMethod]]  
)
FeatureImportances: TypeAlias = list[tuple[str, float]]
FeatureSelector: TypeAlias = (
  Pipeline | BaseEstimator | None
)
HamSpamFeatureImportances: TypeAlias = (
  tuple[FeatureImportances, FeatureImportances]
)
PipelineStep: TypeAlias = tuple[str, BaseEstimator]
PipelineSteps: TypeAlias = Sequence[PipelineStep]
SpamDict: TypeAlias = dict[str, list[str | int]]
TokenizerMethod: TypeAlias = Callable[[str], Tokens]
