"""Implementation of tasks which depend on all other Luigi tasks."""

from typing import override

import luigi

from tasks.ada_boost_task import AdaBoostTask
from tasks.bert_task import BertTask
from tasks.best_bow_task import BestBowTask
from tasks.decision_tree_task import DecisionTreeTask
from tasks.extra_trees_task import ExtraTreesTask
from tasks.gradient_boosting_task import GradientBoostingTask
from tasks.linear_svm_task import LinearSvmTask
from tasks.logistic_regression_task import LogisticRegressionTask
from tasks.naive_bayes_task import NaiveBayesTask
from tasks.nltk_task import NltkTask
from tasks.poly_svm_task import PolySvmTask
from tasks.random_forest_task import RandomForestTask
from tasks.rbf_svm_task import RbfSvmTask
from tasks.stacking_task import StackingTask
from tasks.train_test_split_task import TrainTestSplitTask
from tasks.voting_task import VotingTask


class RootTask(luigi.Task):
  """Depends on all other `Luigi` tasks."""

  @override
  def requires(self):
    return [
      TrainTestSplitTask(),
      NltkTask(),
      NaiveBayesTask(),
      LogisticRegressionTask(),
      DecisionTreeTask(),
      LinearSvmTask(),
      RbfSvmTask(),
      PolySvmTask(),
      RandomForestTask(),
      ExtraTreesTask(),
      AdaBoostTask(),
      GradientBoostingTask(),
      VotingTask(),
      StackingTask(),
      BestBowTask(),
      BertTask(),
    ]
