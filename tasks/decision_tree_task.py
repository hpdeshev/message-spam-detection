"""Implementation of decision tree text classifier."""

from pathlib import Path
from typing import override

import luigi
import optuna
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from common.config import misc
from common.types import  PipelineStep
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from pipeline.utils import get_predictor
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class DecisionTreeClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with a `decision tree` text classifier.

  The classifier can be used for determining feature importances.
  """

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    min_samples_split = trial.suggest_int(
      "min_samples_split", min_samples_leaf+10, min_samples_leaf+20
    )

    return "Decision Tree model", DecisionTreeClassifier(
      min_samples_leaf=min_samples_leaf,
      min_samples_split=min_samples_split,
      random_state=misc().random_seed,
    )


class DecisionTreeTask(luigi.Task):
  """Outputs a `decision tree` text classifier."""

  @override
  def requires(self):
    return {
      "nltk": NltkTask(),
      "train_test_split": TrainTestSplitTask(),
      "feature_estimator": LogisticRegressionTask(),
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
    nltk_task = self.requires()["nltk"]
    builder = DecisionTreeClassifierBuilder(
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
      Path() / "models" / "decision_tree_classifier.pkl"
    )
