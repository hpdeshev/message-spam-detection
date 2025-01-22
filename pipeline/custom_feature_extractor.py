"""Custom feature extraction as part of a Scikit-learn pipeline."""

from collections.abc import Collection

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from common.types import FeatureExtractorMethod, FeatureExtractorMethodData


class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
  """Implements custom feature extraction as part of a `Scikit-learn` pipeline.

  Attributes:
    methods: Callables that provide features extracted from tokenized text.
  """

  def __init__(self, methods: FeatureExtractorMethodData):
    self.methods = methods

  def fit(
    self,
    X: Collection[str],
    y: Collection[int] | None = None
  ) -> "CustomFeatureExtractor":
    return self

  def transform(self, X: Collection[str]) -> np.ndarray[list[int]]:
    return np.array([
      self._extract_features(X, method) for _, method in self.methods
    ]).T

  def get_feature_names_out(
    self,
    input_features: Collection[str] | None = None
  ) -> np.ndarray[str]:
    return np.array([
      method_name for method_name, _ in self.methods
    ])

  def _extract_features(
    self,
    messages: Collection[str],
    method: FeatureExtractorMethod
  ) -> list[int]:
    return [
      method(message.lower().split()) for message in messages
    ]

  # # Support for passing sklearn.utils.estimator_checks.check_estimator.
  # def _more_tags(self):
  #   return {"X_types": ["string"]}
