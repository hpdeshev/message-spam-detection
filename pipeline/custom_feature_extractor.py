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
    methods: Callables that provide features extracted from tokenized text.
  """

  def __init__(self, methods: FeatureExtractorMethodData):
    self.methods = methods

  def fit(
    self,
    X: Collection[str],
    y: Collection[int] | None = None,
  ) -> "CustomFeatureExtractor":
    return self

  def transform(self, X: Collection[str]) -> np.ndarray:
    return np.array([
      self._extract_features(X, method) for _, method in self.methods
    ]).T

  def get_feature_names_out(
    self,
    input_features: Collection[str] | None = None,
  ) -> np.ndarray:
    return np.array([
      method_name for method_name, _ in self.methods
    ])

  def _extract_features(
    self,
    messages: Collection[str],
    method: FeatureExtractorMethod,
  ) -> list[int]:
    return [
      method([token
              for token in re.split(_REGEX_SEPARATORS,
                                    message.lower())
              if token])
      for message in messages
    ]

  # # Support for passing sklearn.utils.estimator_checks.check_estimator.
  # def _more_tags(self):
  #   return {"X_types": ["string"]}
