"""Custom feature extraction as part of a Scikit-learn pipeline."""

from collections.abc import Collection
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from common.config import tokenization
from common.types import FeatureExtractorMethod, FeatureExtractorMethodData


_REGEX_SEPARATORS = str(tokenization().regex_separators)


class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
  """Implements custom feature extraction as part of a `Scikit-learn` pipeline.

  Attributes:
    method_data: Callables that provide features extracted from tokenized text.
  """

  def __init__(self, method_data: FeatureExtractorMethodData):
    self.method_data = method_data

  def fit(
    self,
    X: Collection[str],
    y: Collection[int] | None = None,
  ) -> "CustomFeatureExtractor":
    return self

  def transform(self, X: Collection[str]) -> np.ndarray:
    return np.array([
      self._extract_features(X, method)
      for method in self.method_data.values()
    ]).T

  def get_feature_names_out(
    self,
    input_features: Collection[str] | None = None,
  ) -> np.ndarray:
    return np.array(list(self.method_data))

  def _extract_features(
    self,
    messages: Collection[str],
    method: FeatureExtractorMethod,
  ) -> list[int]:
    return [
      method([token
              for token in re.split(_REGEX_SEPARATORS, message)
              if token])
      for message in messages
    ]

  # # Support for passing sklearn.utils.estimator_checks.check_estimator.
  # def _more_tags(self):
  #   return {"X_types": ["string"]}
