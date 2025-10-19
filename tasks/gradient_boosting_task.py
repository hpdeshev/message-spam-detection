"""Implementation of gradient boosting text classifier."""

from pathlib import Path
from typing import override

import luigi
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from common.config import classification, misc
from common.types import PipelineStep
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from pipeline.utils import get_predictor
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class GradientBoostingClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with a `gradient boosting` text classifier.

  The classifier can be used for determining feature importances.
  """

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    subsample = trial.suggest_float("subsample", 0.1, 1, step=0.1)

    return (
      "Gradient Boosting model", GradientBoostingClassifier(
        learning_rate=0.1, n_estimators=n_estimators,
        subsample=subsample, n_iter_no_change=5,
        validation_fraction=classification().validation_split,
        random_state=misc().random_seed, warm_start=True,
      )
    )


class GradientBoostingTask(luigi.Task):
  """Outputs a `gradient boosting` text classifier."""

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
    builder = GradientBoostingClassifierBuilder(
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
      Path() / "models" / "gradient_boosting_classifier.pkl"
    )
