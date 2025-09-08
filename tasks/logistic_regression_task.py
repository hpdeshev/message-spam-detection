"""Implementation of logistic regression text classifier."""

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
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class LogisticRegressionClassifierBuilder(SgdClassifierBuilder):
  """Builds a pipeline with a `logistic regression` text classifier.
  
  This classifier is a support vector machine (`SVM`) linear model,
  trained with `log loss`.

  The classifier is used by other models for feature selection,
  based on learned weights or dimensionality reduction.
  
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
    else:
      alpha = 1e-4
      eta0 = trial.suggest_float("eta0", 1e-4, 1e-1, step=1e-4)

    return "Logistic Regression model", SGDClassifier(
      loss="log_loss",
      alpha=alpha, learning_rate=learning_rate, eta0=eta0,
      early_stopping=True, n_iter_no_change=5,
      validation_fraction=classification().validation_split,
      shuffle=True, random_state=misc().random_seed,
      warm_start=True,
    )


class LogisticRegressionTask(luigi.Task):
  """Outputs a `logistic regression` text classifier."""

  @override
  def requires(self):
    return {
      "nltk": NltkTask(),
      "train_test_split": TrainTestSplitTask(),
    }

  @override
  def run(self):
    train_df = pd.read_csv(
      self.input()["train_test_split"]["train"].path
    )
    nltk_task = self.requires()["nltk"]
    builder = LogisticRegressionClassifierBuilder(
      context=Context(
        all_names=nltk_task.all_names,
        all_stopwords=nltk_task.all_stopwords,
        stemmer=nltk_task.stemmer,
      )
    )
    builder.build(
      self.output().path, train_df.message, train_df.is_spam
    )

  @override
  def output(self):
    return luigi.LocalTarget(
      Path() / "models" / "logistic_regression_classifier.pkl"
    )
