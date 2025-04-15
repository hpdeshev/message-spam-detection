"""Basic Scikit-learn pipelines for bag-of-words (BoW) classification."""

import os
from pathlib import Path
import pickle
import re

import nltk
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_X_y

from common.config import classification, misc
from common.types import (
  Tokens, FeatureExtractorMethodData,
  FeatureSelector, PipelineStep
)
from pipeline.custom_feature_extractor import CustomFeatureExtractor
from pipeline.utils import (
  get_predictor, get_predictor_name, get_transformers
)
from tasks.utils import get_sample_weights


_CROSS_VAL_FBETA = 0.5  # Spam precision favored.
_CROSS_VAL_SPLITS = 3
_CURRENCY_SYMBOLS = "$£€"
_MIN_DIGITS_IN_NUMBER = 3
_OPTUNA_N_TRIALS = 30
_PATTERN_STARTING_DIGIT = r"^[0-9]?[^\W\d_]+(?:[-'][^\W\d_]+)*$"
_PATTERN_MID_DIGIT = r"^[^\W\d_]+[0-9]?[^\W\d_]*(?:[-'][^\W\d_]+)*$"


def _select_feature_extraction_methods(
  X: pd.Series,
  y: pd.Series,
) -> FeatureExtractorMethodData:
  """Selects a custom `TF-IDF` feature using `_ALL_FEATURE_EXTRACTION_METHODS`.
  
  The custom `TF-IDF` feature is selected based on highest `chi-squared test`
  score with the target label.
  """
  pipe = make_pipeline(
    CustomFeatureExtractor(_ALL_FEATURE_EXTRACTION_METHODS),
    TfidfTransformer(norm=None),  # type: ignore
  )
  features = pipe.fit_transform(X, y)
  feature_selector = SelectKBest(chi2, k="all")  # type: ignore
  feature_selector.fit_transform(features, y)
  print("\n\nK-best scores:")
  print("-" * 80)
  print(feature_selector.scores_)

  return [
    _ALL_FEATURE_EXTRACTION_METHODS[np.argmax(feature_selector.scores_)]
  ]


def _get_currency_count(tokens: Tokens) -> int:
  return sum(1 for token in tokens
             if token[0] in _CURRENCY_SYMBOLS or
                token[-1] in _CURRENCY_SYMBOLS)


def _get_numbers_count(tokens: Tokens) -> int:
  return sum(1 for token in tokens
             if token.isnumeric()
                and len(token) >= _MIN_DIGITS_IN_NUMBER)


def _get_currency_numbers_count(tokens: Tokens) -> int:
  return _get_currency_count(tokens) + _get_numbers_count(tokens)


_ALL_FEATURE_EXTRACTION_METHODS = [
  ("currency_count", _get_currency_count),
  ("numbers_count", _get_numbers_count),
  ("currency_numbers_count", _get_currency_numbers_count),
]


class Context:
  """Context that is passed to `TextClassifierBuilder`.

  Context can, for example, be NLTK corpus data or a feature importance
  estimator.

  Attributes:
    all_names: A set of names from an NLTK corpus.
    all_stopwords: A set of stopwords from an NLTK corpus.
    stemmer: A stemmer for English.
    feature_estimator: Optional, a `BaseEstimator` instance which
      provides some measure for feature importances.
  """

  def __init__(
    self,
    all_names: set[str],
    all_stopwords: set[str],
    stemmer: nltk.PorterStemmer,
    feature_estimator: BaseEstimator | None = None,
  ):
    if all_names is None or all_stopwords is None or stemmer is None:
      raise ValueError(
        f"'all_names', 'all_stopwords' or 'stemmer' is not set in Context."
      )

    self.all_names = all_names
    self.all_stopwords = all_stopwords
    self.stemmer = stemmer
    self.feature_estimator = feature_estimator


class TextClassifierBuilder:
  """Base class for creating trained `BoW` classifier pipelines.

  An instance of this class can be constructed with three parameters:
    - normalized_data: per-document normalization of pipelined
      `TF-IDF` data, normalization is required by some `BoW` classifiers;
    - balanced_weights: if *True*, sample weighting is applied based on
      class frequencies;
    - context: optional, an instance of `Context`.

  No-argument construction implies an already built classifier.
  """

  def __init__(
    self,
    normalized_data: bool = True,
    balanced_weights: bool = True,
    context: Context | None = None,
  ):
    self._normalized_data = normalized_data
    self._balanced_weights = balanced_weights
    self._context = context
    self._study = None

  def build(
    self,
    saved_model_path: Path | str,
    X: pd.Series | None = None,
    y: pd.Series | None = None,
    params: dict[str, bool | float | int | str] | None = None,
  ) -> Pipeline:
    """Creates a trained `BoW` classifier pipeline.
  
    In case of passed `X` and `y`, incremental training is performed if
    `saved_model_path` points to a saved model, otherwise is performed
    cross-validated training and hyperparameter tuning with `Optuna`.

    In order for the `BoW` approach to be applicable, the message spam data
    is tokenized via the `_tokenize` method. Then, `TF-IDF` vectorization is
    performed, including a custom `TF-IDF` feature that is selected based on
    `_ALL_FEATURE_EXTRACTION_METHODS`. `TF-IDF` normalization is not done
    in order to allow for a final normalization after the combination of
    regular and custom `TF-IDF` features.

    A pipeline is created before a cross-validated training process starts:
    - `_create_feature_extractor` is called to create a `TF-IDF` vectorizer;
    - optionally, `_create_feature_selector` is called to create a selector
      for only the most useful `TF-IDF` features;
    - the overridden method `_create_predictor_data` is called in order to
      obtain the pipeline's predictor and its hyperparameters for tuning.

    All pipeline steps but the last predictor one have well-known fixed names.

    There are various improvements in the training process:
    - balanced sample weights are used, based on the distribution of classes;
    - `F-beta score` for the spam class is chosen as the metric to
      tune hyperparameters on, because it provides a convenient means of
      tweaking the relative importance of `precision` vs. `recall` and is
      preferrable to the `accuracy` metric for this project's case of
      imbalanced data labels. For the spam class it seems `precision`
      is the more important metric when compared to `recall`, because
      incorrectly flagging a regular message as spam may cause a user
      to not see the message on time or even miss that message. So the
      `F-beta score` is set to favor `precision` over `recall`;
    - a cross-validation iteration is skipped if no hyperparameters
      are provided.

    To perform the actual training process, `build` calls `_fit`.
    Finally, the built model is saved.

    Args:
      saved_model_path: Path used for saving and loading the model.
      X: Optional, text data used for training.
      y: Optional, labels used for training.
      params: Optional, a classifier-specific parameter-value dictionary.

    Returns:
      A trained `BoW` classifier pipeline.

    Raises:
      ValueError: Various input arguments are incorrectly
      passed as *None*.
    """
    model = None
    if Path.exists(Path(saved_model_path)):
      model = pickle.load(open(saved_model_path, "rb"))
      if X is None or y is None:
        return model

    if X is None or y is None:
      raise ValueError("X and/or y is None.")
    X, y = check_X_y(X, y, dtype=None, ensure_2d=False)  # type: ignore

    if model is not None:
      self._fit(model, X, y,  # type: ignore
                balanced_weights=self._balanced_weights,
                incremental=True, verbose=True,
                params=params)
      try:
        pickle.dump(model, open(saved_model_path, "wb"))
      except Exception as e:
        print("pickle.dump:", e, sep="\n")
        os.remove(saved_model_path)
      return model

    sampler = optuna.samplers.TPESampler(
      seed=misc().random_seed  # type: ignore
    )
    self._study = optuna.create_study(
      direction="maximize", sampler=sampler
    )
    feature_extraction_methods = _select_feature_extraction_methods(
      X, y  # type: ignore
    )

    tried_models = {}
    def mean_score(trial: optuna.Trial) -> float:
      if self._context is None:
        raise ValueError(f"Context is None.")
      estimators = [(
        "Features",
        FeatureUnion([
          ("TF-IDF",
           self._create_feature_extractor(trial=trial)),  # type: ignore
          (
            "Custom TF-IDF",
            make_pipeline(
              CustomFeatureExtractor(feature_extraction_methods),
              TfidfTransformer(norm=None),  # type: ignore
            )
          )
        ])
      )]
      if self._normalized_data:
        estimators += [("Normalizer", Normalizer(copy=False))]
      if self._context.feature_estimator is not None:
        estimators += [("Feature Selector", self._create_feature_selector(trial=trial))]
      estimators += [self._create_predictor_data(trial=trial)]
      classifier = Pipeline(estimators)

      # If no params to optimize - exit without cross-validation.
      if len(trial.params) == 0:
        if trial.number == 0:
          tried_models[0] = classifier
        return 0

      if self._balanced_weights:
        predictor_name = get_predictor_name(classifier)
        params = {f"{predictor_name}__sample_weight":
                    get_sample_weights(y)}  # type: ignore
      else:
        params = None
      tried_models[trial.number] = classifier
      scores = cross_val_score(
        classifier, X, y,
        scoring=make_scorer(fbeta_score, beta=_CROSS_VAL_FBETA),
        cv=StratifiedKFold(_CROSS_VAL_SPLITS),
        params=params,  # type: ignore
        n_jobs=-1, verbose=4,
      )
      return scores.mean()

    self._study.optimize(mean_score, n_trials=_OPTUNA_N_TRIALS,
                         n_jobs=1, show_progress_bar=True)
    model = tried_models[self._study.best_trial.number]
    model = self._fit(model, X, y,  # type: ignore
                      balanced_weights=self._balanced_weights, verbose=True,
                      params=params)
    try:
      pickle.dump(model, open(saved_model_path, "wb"))
    except Exception as e:
      print("pickle.dump:", e, sep="\n")
      os.remove(saved_model_path)
    return model

  def _fit(
    self,
    model: Pipeline,
    X: pd.Series,
    y: pd.Series,
    balanced_weights: bool = True,
    incremental: bool = False,
    verbose: bool = False,
    params: dict[str, bool | float | int | str] | None = None,
  ) -> Pipeline:
    transformers = get_transformers(model)
    if incremental:
      X_tfidf = transformers.transform(X)
    else:
      X_tfidf = transformers.fit_transform(X, y)

    predictor = get_predictor(model)
    params = params if params is not None else {}
    predictor.set_params(**params)

    sample_weights = get_sample_weights(y) if balanced_weights else None
    predictor.fit(X_tfidf, y, sample_weight=sample_weights)  # type: ignore
    if verbose:
      y_pred = predictor.predict(X_tfidf)  # type: ignore
      print(f"train_accuracy={accuracy_score(y, y_pred):.3f}")
    return model

  def _create_feature_extractor(self, trial: optuna.Trial) -> TfidfVectorizer:
    return TfidfVectorizer(
      lowercase=False, tokenizer=self._tokenize,
      token_pattern=None, ngram_range=(1, 2),  # type: ignore
      min_df=1e-3, max_df=0.5,  # type: ignore
      norm=None,  # type: ignore
    )

  def _create_feature_selector(self, trial: optuna.Trial) -> FeatureSelector:
    if self._context is None:
      raise ValueError(f"Context is None.")
    feature_selector_type = classification().feature_selector_type
    if feature_selector_type == "model":
      return SelectFromModel(self._context.feature_estimator, threshold=1e-5)
    elif feature_selector_type == "svd":
      n_components = trial.suggest_int(
        "n_components",
        1, min(self._context.feature_estimator.n_features_in_, 1000),  # type: ignore
      )
      return make_pipeline(
        TruncatedSVD(n_components=n_components,
                     random_state=misc().random_seed),  # type: ignore
        Normalizer(copy=False),
      )
    else:
      raise ValueError(
        f"Invalid feature selector type '{feature_selector_type}'."
      )

  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    return PipelineStep()

  def _tokenize(self, message: str) -> list[str]:
    if self._context is None:
      raise ValueError(f"Context is None.")
    tokens = message.lower().split()
    return [
      self._context.stemmer.stem(token)
      for token in tokens
      if (token not in self._context.all_names
          and token not in self._context.all_stopwords
          and (re.search(_PATTERN_STARTING_DIGIT, token, re.U) is not None
               or re.search(_PATTERN_MID_DIGIT, token, re.U) is not None))
    ]
