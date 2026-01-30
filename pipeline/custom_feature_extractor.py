"""Custom feature extraction as part of a Scikit-learn pipeline."""

from collections.abc import Iterable
import re

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin

from common.config import tokenization
from pipeline.feature_detectors import FeatureDetectorBase


_REGEX_SEPARATORS = tokenization().regex_separators


class CustomFeatureExtractor(TransformerMixin, BaseEstimator):
  """Implements custom feature extraction as part of a `Scikit-learn` pipeline.

  Attributes:
    detectors: `pipeline.feature_detectors.FeatureDetectorBase` instances that
      detect features in tokenized text.
  """

  def __init__(self, detectors: Iterable[FeatureDetectorBase]):
    self.detectors = detectors

  def fit(
    self,
    X: ArrayLike,
    y: ArrayLike | None = None,
  ) -> "CustomFeatureExtractor":
    return self

  def transform(self, X: Iterable[str]) -> np.ndarray:
    return np.array([
      self._extract_features(X, detector)
      for detector in self.detectors
    ]).T

  def get_feature_names_out(
    self,
    input_features: ArrayLike | None = None,
  ) -> np.ndarray:
    return np.array([detector.feature_name
                     for detector in self.detectors])

  def _extract_features(
    self,
    messages: Iterable[str],
    detector: FeatureDetectorBase,
  ) -> list[int]:
    return [
      detector.count(token
                     for token in re.split(_REGEX_SEPARATORS, message)
                     if token)
      for message in messages
    ]

  # # Support for passing sklearn.utils.estimator_checks.check_estimator.
  # def _more_tags(self):
  #   return {"X_types": ["string"]}
