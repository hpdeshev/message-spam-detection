"""Implementation of voting text classifier."""

from collections.abc import Sequence
import itertools
from pathlib import Path
from typing import override

import luigi
import optuna
import pandas as pd
from sklearn.ensemble import VotingClassifier

from common.types import PipelineStep, PipelineSteps
from tasks.extra_trees_task import \
  ExtraTreesClassifierBuilder, ExtraTreesTask
from tasks.linear_svm_task import \
  LinearSvmClassifierBuilder, LinearSvmTask
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.poly_svm_task import \
  PolySvmClassifierBuilder, PolySvmTask
from tasks.random_forest_task import \
  RandomForestClassifierBuilder, RandomForestTask
from tasks.rbf_svm_task import \
  RbfSvmClassifierBuilder, RbfSvmTask
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from pipeline.utils import get_predictor, get_predictor_data
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class VotingClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with a `voting` text classifier."""

  def __init__(
    self,
    estimators: Sequence[PipelineSteps] | None = None,
    **params,
  ):
    super().__init__(**params)
    self._estimators = estimators

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    if not self._estimators:
      raise ValueError("No estimators have been configured.")
    estimators_index = trial.suggest_int(
      "estimators_index", 0, len(self._estimators) - 1
    )
    return "Voting model", VotingClassifier(
      estimators=list(self._estimators[estimators_index]),
      voting="hard", n_jobs=1,
    )


class VotingTask(luigi.Task):
  """Outputs a `voting` text classifier."""

  @override
  def requires(self):
    return {
      "nltk": NltkTask(),
      "train_test_split": TrainTestSplitTask(),
      "feature_estimator": LogisticRegressionTask(),
      "linear_svm": LinearSvmTask(),
      "rbf_svm": RbfSvmTask(),
      "poly_svm": PolySvmTask(),
      "random_forest": RandomForestTask(),
      "extra_trees": ExtraTreesTask(),
    }

  @override
  def run(self):
    train_df = pd.read_csv(
      self.input()["train_test_split"]["train"].path
    )
    logistic_regression_builder = LogisticRegressionClassifierBuilder()
    logistic_regression_classifier = logistic_regression_builder.build(
      self.input()["feature_estimator"].path
    )
    linear_svm_builder = LinearSvmClassifierBuilder()
    linear_svm_classifier = linear_svm_builder.build(
      self.input()["linear_svm"].path
    )
    rbf_svm_builder = RbfSvmClassifierBuilder()
    rbf_svm_classifier = rbf_svm_builder.build(
      self.input()["rbf_svm"].path
    )
    poly_svm_builder = PolySvmClassifierBuilder()
    poly_svm_classifier = poly_svm_builder.build(
      self.input()["poly_svm"].path
    )
    random_forest_builder = RandomForestClassifierBuilder()
    random_forest_classifier = random_forest_builder.build(
      self.input()["random_forest"].path
    )
    extra_trees_builder = ExtraTreesClassifierBuilder()
    extra_trees_classifier = extra_trees_builder.build(
      self.input()["extra_trees"].path
    )

    voting_estimators = list(
      itertools.combinations([
        get_predictor_data(linear_svm_classifier),
        get_predictor_data(rbf_svm_classifier),
        get_predictor_data(poly_svm_classifier),
        get_predictor_data(random_forest_classifier),
        get_predictor_data(extra_trees_classifier),
      ], r=3)
    )

    nltk_task = self.requires()["nltk"]
    builder = VotingClassifierBuilder(
      estimators=voting_estimators,
      context=Context(
        all_names=nltk_task.all_names,
        all_stopwords=nltk_task.all_stopwords,
        stemmer=nltk_task.stemmer,
        feature_estimator=get_predictor(logistic_regression_classifier),
      )
    )
    builder.build(
      self.output().path, train_df.message, train_df.is_spam
    )

  @override
  def output(self):
    return luigi.LocalTarget(
      Path() / "models" / "voting_classifier.pkl"
    )
