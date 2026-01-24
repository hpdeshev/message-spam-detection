"""General types."""

from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class SpamData(TypedDict):
  message: list[str]
  kind: list[str]
  is_spam: list[int]

type ClassWeights = dict[int, float]
type Tokens = Iterable[str]
type FeatureExtractor = BaseEstimator
type FeatureExtractorMethod = Callable[[Tokens], int]
type FeatureExtractorMethodData = dict[str, FeatureExtractorMethod]
type FeatureImportances = list[tuple[str, float]]
type FeatureSelector = Pipeline | BaseEstimator | None
type HamSpamFeatureImportances = tuple[FeatureImportances, FeatureImportances]
type PipelineStep = tuple[str, BaseEstimator]
type PipelineSteps = Sequence[PipelineStep]
