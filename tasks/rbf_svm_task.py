"""A Radial Basis Function (RBF) support vector machine (SVM) builder."""

from pathlib import Path

import luigi
import optuna
import pandas as pd
from sklearn.svm import SVC
from typing_extensions import override

from common.config import misc
from common.types import PipelineStep
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from pipeline.utils import get_predictor
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class RbfSvmClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with an `RBF SVM` text classifier."""

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    C = trial.suggest_float("C", 1e-2, 1e2, step=1e-2)
    gamma = trial.suggest_float("gamma", 1e-2, 1, step=1e-2)

    return "RBF SVM model", SVC(
      C=C, kernel="rbf", gamma=gamma,
      probability=False, random_state=misc().random_seed,
    )


class RbfSvmTask(luigi.Task):
  """Outputs an `RBF SVM` text classifier."""

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
    builder = RbfSvmClassifierBuilder(
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
      Path() / "models" / "rbf_svm_classifier.pkl"
    )
