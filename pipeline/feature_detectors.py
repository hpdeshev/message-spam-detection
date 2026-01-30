"""Feature detection as part of a Scikit-learn pipeline."""

from abc import ABC, abstractmethod

from common.types import Token, Tokens


class FeatureDetectorBase(ABC):
  def __init__(self, feature_name: str):
    if not feature_name:
      raise ValueError("Non-empty feature name must be provided.")
    self._feature_name = feature_name

  @property
  def feature_name(self) -> str:
    return self._feature_name

  def count(self, tokens: Tokens) -> int:
    return sum(1 for token in tokens if self.check(token))

  @abstractmethod
  def check(self, token: Token) -> bool:
    pass


class CompositeFeatureDetector(FeatureDetectorBase):
  def __init__(
    self,
    feature_name: str,
    components: tuple[FeatureDetectorBase, ...],
  ):
    if not components:
      raise ValueError("At least one detector must be provided.")
    super().__init__(feature_name)
    self._components = components

  def check(self, token: Token) -> bool:
    return any(detector.check(token) for detector in self._components)


class CurrencyDetector(FeatureDetectorBase):
  def __init__(self, feature_name: str, currency_symbols: str):
    if not currency_symbols:
      raise ValueError("Non-empty currency symbols must be provided.")
    super().__init__(feature_name)
    self._currency_symbols = currency_symbols

  def check(self, token: Token) -> bool:
    if not token:
      return False
    if len(token) == 1 and token in self._currency_symbols:
      return True
    return (token[1:].isnumeric() if token[0] in self._currency_symbols
            else token[:-1].isnumeric() if token[-1] in self._currency_symbols
            else False)


class NumberDetector(FeatureDetectorBase):
  def __init__(self, feature_name: str, min_digits: int):
    if min_digits <= 0:
      raise ValueError("Minimal digit count must be a positive number.")
    super().__init__(feature_name)
    self._min_digits = min_digits

  def check(self, token: Token) -> bool:
    if not token:
      return False
    return token.isnumeric() and len(token) >= self._min_digits
