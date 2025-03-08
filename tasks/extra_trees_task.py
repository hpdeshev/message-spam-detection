"""Implementation of extra trees text classifier."""

from pathlib import Path

import luigi
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from typing_extensions import override

from common.config import misc
from common.types import PipelineStep
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from pipeline.utils import get_predictor
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class ExtraTreesClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with an `extra trees` text classifier.

  The classifier can be used for determining feature importances.
  """

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    return "Extra-trees model", ExtraTreesClassifier(
      n_estimators=n_estimators, oob_score=False,
      n_jobs=-1, random_state=misc().random_seed,
      warm_start=True,
    )

class ExtraTreesTask(luigi.Task):
  """Outputs an `extra trees` text classifier."""

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
      self.input()["train_test_split"]["train"].path, index_col=0
    )
    logistic_regression_builder = LogisticRegressionClassifierBuilder()
    logistic_regression_classifier = logistic_regression_builder.build(
      self.input()["feature_estimator"].path
    )
    nltk_task = self.requires()["nltk"]
    builder = ExtraTreesClassifierBuilder(
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
      Path() / "models" / "extra_trees_classifier.pkl"
    )
