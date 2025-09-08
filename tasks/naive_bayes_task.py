"""Implementation of naive Bayes text classifier."""

from pathlib import Path

import luigi
import optuna
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from typing_extensions import override

from common.types import PipelineStep
from pipeline.text_classifier_builder import Context, TextClassifierBuilder
from tasks.nltk_task import NltkTask
from tasks.train_test_split_task import TrainTestSplitTask


class NaiveBayesClassifierBuilder(TextClassifierBuilder):
  """Builds a pipeline with a `naive Bayes` text classifier.

  Normalization and balanced sample weights are not used because
  `naive Bayes` handles class and sample frequencies as part of its algorithm.

  The classifier can be used for determining ham/spam feature importances.
  """

  def __init__(self, context: Context | None = None):
    super().__init__(
      normalized_data=False, balanced_weights=False, context=context
    )

  @override
  def _create_predictor_data(self, trial: optuna.Trial) -> PipelineStep:
    return "Naive Bayes model", MultinomialNB(
      alpha=1, force_alpha=True
    )


class NaiveBayesTask(luigi.Task):
  """Outputs a `naive Bayes` text classifier."""

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
    builder = NaiveBayesClassifierBuilder(
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
      Path() / "models" / "naive_bayes_classifier.pkl"
    )
