"""Utilities supporting implementation of Luigi tasks."""

from collections.abc import Collection

import numpy as np
from sklearn.utils.class_weight import (
  compute_class_weight, compute_sample_weight
)

from common.types import ClassWeights


def get_class_weights(y: Collection[int]) -> ClassWeights:
  """Returns class weights inversely proportional to class frequencies.
  
  Args:
    y: Spam data labels.

  Returns:
    Class weights.
  """
  return dict(enumerate(compute_class_weight(
    class_weight="balanced", classes=np.unique(y), y=y
  )))


def get_sample_weights(y: Collection[int]) -> np.ndarray:
  """Returns sample weights inversely proportional to class frequencies.
  
  Args:
    y: Spam data labels.

  Returns:
    Sample weights.
  """
  return compute_sample_weight(
    class_weight=get_class_weights(y), y=y
  )
