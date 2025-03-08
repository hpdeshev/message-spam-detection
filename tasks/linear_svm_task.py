"""Implementation of linear support vector machine (SVM) text classifier."""

from pathlib import Path

import luigi
import optuna
import pandas as pd
from sklearn.linear_model import SGDClassifier
from typing_extensions import override

from common.config import classification, misc
from common.types import PipelineStep
from pipeline.sgd_classifier_builder import SgdClassifierBuilder
from pipeline.text_classifier_builder import Context
from pipeline.utils import get_predictor
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class LinearSvmClassifierBuilder(SgdClassifierBuilder):
  """Builds a pipeline with a `linear SVM` text classifier.
  
  This classifier is a `linear SVM` model, trained with `hinge loss`.
  The classifier can be used for determining ham/spam feature importances.
  """

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    learning_rate = trial.suggest_categorical(
      "learning_rate",
      ["constant", "optimal", "invscaling", "adaptive"],
    )
    if learning_rate == "optimal":
      alpha = trial.suggest_float("alpha", 1e-4, 1e-1, step=1e-4)
      eta0 = 0.0
    elif learning_rate == "invscaling":
      alpha = 1e-4
      eta0 = trial.suggest_float("eta0", 1e1, 1e3, step=1)
    else:
      alpha = 1e-4
      eta0 = trial.suggest_float("eta0", 1e-4, 1e-1, step=1e-4)

    return "Linear SVM model", SGDClassifier(
      loss="hinge",
      alpha=alpha, learning_rate=learning_rate, eta0=eta0,
      early_stopping=True, n_iter_no_change=5,
      validation_fraction=classification().validation_split,
      shuffle=True, random_state=misc().random_seed,
      warm_start=True,
    )


class LinearSvmTask(luigi.Task):
  """Outputs a `linear SVM` text classifier."""

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
    builder = LinearSvmClassifierBuilder(
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
      Path() / "models" / "linear_svm_classifier.pkl"
    )
