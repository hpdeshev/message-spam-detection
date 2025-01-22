"""Selection of the best bag-of-words (BoW) classifier for spam data."""

from collections.abc import Collection
import os
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, euclidean_distances
from typing_extensions import override

from common.types import ClassificationReport
from pipeline.text_classifier_builder import TextClassifierBuilder
from tasks.ada_boost_task import \
  AdaBoostClassifierBuilder, AdaBoostTask
from tasks.decision_tree_task import \
  DecisionTreeClassifierBuilder, DecisionTreeTask
from tasks.extra_trees_task import \
  ExtraTreesClassifierBuilder, ExtraTreesTask
from tasks.gradient_boosting_task import \
  GradientBoostingClassifierBuilder, GradientBoostingTask
from tasks.linear_svm_task import \
  LinearSvmClassifierBuilder, LinearSvmTask
from tasks.logistic_regression_task import \
  LogisticRegressionClassifierBuilder, LogisticRegressionTask
from tasks.naive_bayes_task import \
  NaiveBayesClassifierBuilder, NaiveBayesTask
from tasks.poly_svm_task import \
  PolySvmClassifierBuilder, PolySvmTask
from tasks.random_forest_task import \
  RandomForestClassifierBuilder, RandomForestTask
from tasks.rbf_svm_task import \
  RbfSvmClassifierBuilder, RbfSvmTask
from tasks.stacking_task import \
  StackingClassifierBuilder, StackingTask
from tasks.voting_task import \
  VotingClassifierBuilder, VotingTask
from tasks.train_test_split_task import TrainTestSplitTask


class BestBowTask(luigi.Task):
  """Outputs the best `BoW` spam classifier.
  
  Selection method:
  1. The following metrics are collected for all `BoW` classifiers,
     listed by order of priority:
     - spam `precision`
     - spam `F1-score`
     - `accuracy`;
  2. The best metrics are determined and for every classifier is
     calculated the Euclidean distance from these best metrics;
  3. Based on Euclidean distance, the top 3 classifiers are selected;
  4. Finally, the best classifier is selected after additionally
     sorting the metric scores by order of priority, as listed in step 1.

  The best `BoW` classifier is indicated by a symbolic link to its .pkl file.
  """

  @override
  def requires(self):
    return {
      "train_test_split" : TrainTestSplitTask(),
      "ada_boost" : AdaBoostTask(),
      "decision_tree" : DecisionTreeTask(),
      "extra_trees" : ExtraTreesTask(),
      "gradient_boosting" : GradientBoostingTask(),
      "naive_bayes" : NaiveBayesTask(),
      "linear_svm" : LinearSvmTask(),
      "logistic_regression" : LogisticRegressionTask(),
      "rbf_svm" : RbfSvmTask(),
      "poly_svm" : PolySvmTask(),
      "random_forest" : RandomForestTask(),
      "stacking" : StackingTask(),
      "voting" : VotingTask(),
    }

  @override
  def run(self):
    test_df = pd.read_csv(
      self.input()["train_test_split"]["test"].path, index_col=0
    )
    ada_boost_scores = self._get_scores(
      AdaBoostClassifierBuilder(), "ada_boost",
      test_df.message, test_df.is_spam
    )
    decision_tree_scores = self._get_scores(
      DecisionTreeClassifierBuilder(), "decision_tree",
      test_df.message, test_df.is_spam
    )
    extra_trees_scores = self._get_scores(
      ExtraTreesClassifierBuilder(), "extra_trees",
      test_df.message, test_df.is_spam
    )
    gradient_boosting_scores = self._get_scores(
      GradientBoostingClassifierBuilder(), "gradient_boosting",
      test_df.message, test_df.is_spam
    )
    naive_bayes_scores = self._get_scores(
      NaiveBayesClassifierBuilder(), "naive_bayes",
      test_df.message, test_df.is_spam
    )
    linear_svm_scores = self._get_scores(
      LinearSvmClassifierBuilder(), "linear_svm",
      test_df.message, test_df.is_spam
    )
    logistic_regression_scores = self._get_scores(
      LogisticRegressionClassifierBuilder(), "logistic_regression",
      test_df.message, test_df.is_spam
    )
    rbf_svm_scores = self._get_scores(
      RbfSvmClassifierBuilder(), "rbf_svm",
      test_df.message, test_df.is_spam
    )
    poly_svm_scores = self._get_scores(
      PolySvmClassifierBuilder(), "poly_svm",
      test_df.message, test_df.is_spam
    )
    random_forest_scores = self._get_scores(
      RandomForestClassifierBuilder(), "random_forest",
      test_df.message, test_df.is_spam
    )
    stacking_scores = self._get_scores(
      StackingClassifierBuilder(), "stacking",
      test_df.message, test_df.is_spam
    )
    voting_scores = self._get_scores(
      VotingClassifierBuilder(), "voting",
      test_df.message, test_df.is_spam
    )

    classifier_names = [
      "stacking", "voting",
      "gradient_boosting", "ada_boost",
      "extra_trees", "random_forest",
      "poly_svm", "rbf_svm", "linear_svm",
      "decision_tree", "logistic_regression",
      "naive_bayes"
    ]

    classifier_scores = [
      stacking_scores, voting_scores,
      gradient_boosting_scores, ada_boost_scores,
      extra_trees_scores, random_forest_scores,
      poly_svm_scores, rbf_svm_scores, linear_svm_scores,
      decision_tree_scores, logistic_regression_scores,
      naive_bayes_scores
    ]

    classifier_scores_df = pd.DataFrame({
      "classifier_name" : classifier_names,
      "spam_precision" : [score["Spam"]["precision"] for score in classifier_scores],
      "spam_f1" : [score["Spam"]["f1-score"] for score in classifier_scores],
      "accuracy" : [score["accuracy"] for score in classifier_scores],
    })

    euclidean_dist_ds = classifier_scores_df.apply(
      lambda row: euclidean_distances(
        [[row.spam_precision, row.spam_f1, row.accuracy]],
        [[classifier_scores_df.spam_precision.max(),
          classifier_scores_df.spam_f1.max(),
          classifier_scores_df.accuracy.max()]]
      )[0],
      axis=1
    )
    euclidean_dist_ds.name = "euclidean_dist"

    final_classifier_scores_df = pd.concat(
      [classifier_scores_df, euclidean_dist_ds], axis=1
    )
    final_classifier_scores_df = final_classifier_scores_df.sort_values(
      by="euclidean_dist", ascending=True
    )
    final_classifier_scores_df = final_classifier_scores_df.head(3)
    final_classifier_scores_df = final_classifier_scores_df.sort_values(
      by=["spam_precision", "spam_f1", "accuracy"], ascending=False
    )
    best_bow_classifier_name = (final_classifier_scores_df.head(1)
                                .classifier_name.values[0])

    _save_classifier_scores(
      classifier_scores_df,
      self.output()["bow_classifier_scores"].path
    )
    os.symlink(
      os.path.abspath(self.input()[best_bow_classifier_name].path),
      self.output()["best_bow_classifier"].path
    )

  @override
  def output(self):
    return {
      "bow_classifier_scores" :
        luigi.LocalTarget(Path() / "figures" / "bow_classifier_scores.png"),
      "best_bow_classifier" :
        luigi.LocalTarget(Path() / "models" / "best_bow_classifier.pkl"),
    }

  def _get_scores(
    self,
    builder: TextClassifierBuilder,
    input_name: str,
    X: Collection[str],
    y: Collection[int]
  ) -> ClassificationReport:
    classifier = builder.build(
      self.input()[input_name].path
    )
    return classification_report(
      y, classifier.predict(X),
      labels=[0, 1],
      target_names=["Ham", "Spam"],
      digits=3, output_dict=True,
      zero_division=np.nan
    )


def _save_classifier_scores(
  classifier_scores_df: pd.DataFrame,
  filename: str
) -> None:
  score_names = ["spam precision", "spam F1-score", "accuracy"]
  scores = [classifier_scores_df.spam_precision,
            classifier_scores_df.spam_f1,
            classifier_scores_df.accuracy]
  bar_size = 0.25
  padding = 0.75
  y_locs = (
    np.arange(len(classifier_scores_df.classifier_name))
    * (len(scores) * bar_size + padding)
  )
  _, ax = plt.subplots(figsize=(12, 10))
  for i, score_name in enumerate(score_names):
    ax.barh(
      y_locs + ((len(score_names) // 2) - i) * bar_size,
      scores[i], height=bar_size, label=score_name,
    )
  ax.set(
    yticks=y_locs,
    yticklabels=classifier_scores_df.classifier_name,
    ylim=[
      0 - len(score_names) * bar_size,
      (len(classifier_scores_df.classifier_name)
       * (len(score_names) * bar_size + padding)
       - len(score_names) * bar_size),
    ],
  )

  scores_threshold = 0.96
  ax.set_xticks(np.arange(0, 1.2, 0.2).tolist() + [scores_threshold])
  ax.vlines(scores_threshold,
            ax.get_ylim()[0], ax.get_ylim()[1],
            linestyles="dotted", colors="black")
  ax.set_title("Performance scores of various BoW classifiers")
  ax.legend(loc="upper left")
  plt.savefig(fname=filename)
