"""General types."""

from collections.abc import Iterable, Sequence
from typing import TypedDict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class SpamData(TypedDict):
  message: list[str]
  kind: list[str]
  is_spam: list[int]

type ClassWeights = dict[int, float]
type Token = str
type Tokens = Iterable[Token]
type FeatureExtractor = BaseEstimator
type FeatureImportances = list[tuple[str, float]]
type FeatureSelector = Pipeline | BaseEstimator | None
type HamSpamFeatureImportances = tuple[FeatureImportances, FeatureImportances]
type PipelineStep = tuple[str, BaseEstimator]
type PipelineSteps = Sequence[PipelineStep]
