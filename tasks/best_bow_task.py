"""Selection of the best bag-of-words (BoW) classifier for spam data."""

import os
from pathlib import Path
from typing import override

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, euclidean_distances

from common.types import ClassificationReport
from pipeline.text_classifier_builder import TextClassifierBuilder
from tasks.ada_boost_task import AdaBoostTask
from tasks.decision_tree_task import DecisionTreeTask
from tasks.extra_trees_task import ExtraTreesTask
from tasks.gradient_boosting_task import GradientBoostingTask
from tasks.linear_svm_task import LinearSvmTask
from tasks.naive_bayes_task import NaiveBayesTask
from tasks.poly_svm_task import PolySvmTask
from tasks.random_forest_task import RandomForestTask
from tasks.rbf_svm_task import RbfSvmTask
from tasks.stacking_task import StackingTask
from tasks.voting_task import VotingTask
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
     calculated an Euclidean distance from these best metrics;
  3. Based on the Euclidean distance, the top 3 classifiers are selected;
  4. Finally, the best classifier is selected after additionally
     sorting the metric scores by order of priority, as listed in step 1.

  The best `BoW` classifier is indicated by a symbolic link to its .pkl file.
  """

  @override
  def requires(self):
    return {
      "train_test_split": TrainTestSplitTask(),
      "ada_boost": AdaBoostTask(),
      "decision_tree": DecisionTreeTask(),
      "extra_trees": ExtraTreesTask(),
      "gradient_boosting": GradientBoostingTask(),
      "naive_bayes": NaiveBayesTask(),
      "linear_svm": LinearSvmTask(),
      "rbf_svm": RbfSvmTask(),
      "poly_svm": PolySvmTask(),
      "random_forest": RandomForestTask(),
      "stacking": StackingTask(),
      "voting": VotingTask(),
    }

  @override
  def run(self):
    test_df = pd.read_csv(
      self.input()["train_test_split"]["test"].path
    )

    classifier_scores: dict[str, ClassificationReport] = {}
    for name, _ in list(self.requires().items())[1:]:
      classifier_scores[name] = self._get_scores(
        name, test_df.message, test_df.is_spam
      )

    classifier_scores_df = pd.DataFrame({
      "classifier_name": classifier_scores.keys(),
      "spam_precision":
        [score["Spam"]["precision"] for score in classifier_scores.values()],
      "spam_f1":
        [score["Spam"]["f1-score"] for score in classifier_scores.values()],
      "accuracy":
        [score["accuracy"] for score in classifier_scores.values()],
    })

    max_scores_2d = np.array([[
      classifier_scores_df.spam_precision.max(),
      classifier_scores_df.spam_f1.max(),
      classifier_scores_df.accuracy.max()
    ]])
    euclidean_dist_ds = classifier_scores_df.apply(
      lambda row: euclidean_distances(
        X=np.array([[row.spam_precision, row.spam_f1, row.accuracy]]),
        Y=max_scores_2d, squared=True,
      )[0, 0],
      axis=1,
    )
    euclidean_dist_ds.name = "euclidean_dist_max"

    final_classifier_scores_df = pd.concat(
      [classifier_scores_df, euclidean_dist_ds], axis=1
    )
    final_classifier_scores_df = final_classifier_scores_df.sort_values(
      by="euclidean_dist_max", ascending=True
    )
    final_classifier_scores_df = final_classifier_scores_df.head(3)
    final_classifier_scores_df = final_classifier_scores_df.sort_values(
      by=["spam_precision", "spam_f1", "accuracy"], ascending=False
    )

    _save_classifier_scores(
      classifier_scores_df,
      self.output()["bow_classifier_scores"].path,
    )
    best_bow_classifier_name = (final_classifier_scores_df.head(1)
                                .classifier_name.values[0])
    os.symlink(
      os.path.abspath(self.input()[best_bow_classifier_name].path),
      self.output()["best_bow_classifier"].path,
    )

  @override
  def output(self):
    return {
      "bow_classifier_scores":
        luigi.LocalTarget(Path() / "figures" / "bow_classifier_scores.png"),
      "best_bow_classifier":
        luigi.LocalTarget(Path() / "models" / "best_bow_classifier.pkl"),
    }

  def _get_scores(
    self,
    input_name: str,
    X: pd.Series,
    y: pd.Series,
  ) -> ClassificationReport:
    classifier = TextClassifierBuilder().build(
      self.input()[input_name].path
    )
    return classification_report(
      y, classifier.predict(X),
      labels=[0, 1],
      target_names=["Ham", "Spam"],
      digits=3, output_dict=True,
      zero_division=np.nan,
    )


def _save_classifier_scores(
  classifier_scores_df: pd.DataFrame,
  filename: str,
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
      scores[i][::-1], height=bar_size, label=score_name,
    )
  ax.set(
    yticks=y_locs,
    yticklabels=classifier_scores_df.classifier_name[::-1],
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
