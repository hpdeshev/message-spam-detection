"""Message dataset formation and splitting into training and test subsets."""

from pathlib import Path

import luigi
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import override

from common.config import misc
from tasks.email_preprocess_task import EmailPreprocessTask
from tasks.sms_preprocess_task import SmsPreprocessTask


class TrainTestSplitTask(luigi.Task):
  """Outputs a training set and a test set out of a message dataset.
  
  Depending on email and SMS enablement settings in `luigi.cfg`,
  a message dataset is formed from email and/or SMS spam data.

  Based on another setting in `luigi.cfg`, the message dataset is further
  split into a part used for training and a part used for testing.

  One more setting in `luigi.cfg` specifies whether duplicates in the
  spam data are removed.

  Label-based stratification is applied to reflect imbalanced ham-spam
  message distribution.
  """

  duplicates = luigi.BoolParameter(False)
  email = luigi.BoolParameter(True)
  sms = luigi.BoolParameter(False)
  test_split = luigi.FloatParameter(0.2)

  @override
  def requires(self):
    deps = {}
    if self.email:
      deps["email"] = EmailPreprocessTask()
    if self.sms:
      deps["sms"] = SmsPreprocessTask()
    return deps

  @override
  def run(self):
    datasets = []
    if self.email:
      email_spam_df = pd.read_csv(self.input()["email"].path, index_col=0)
      if not self.duplicates:
        email_spam_df = email_spam_df.drop_duplicates()
      datasets += [email_spam_df]
    if self.sms:
      sms_spam_df = pd.read_csv(self.input()["sms"].path, index_col=0)
      if not self.duplicates:
        sms_spam_df = sms_spam_df.drop_duplicates()
      datasets += [sms_spam_df]
    spam_df = pd.concat(datasets)
    if not self.duplicates:
      spam_df = spam_df.drop_duplicates()

    (X_train, X_test,
     y_train, y_test) = train_test_split(
      spam_df[["message", "type"]], spam_df.is_spam,
      test_size=self.test_split,
      random_state=misc().random_seed, shuffle=True,
      stratify=spam_df.is_spam
    )

    output = self.output()
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv(output["train"].path)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(output["test"].path)

  @override
  def output(self):
    return {
      "train" : luigi.LocalTarget(
        Path() / "data" / "train_messages.csv"
      ),
      "test" : luigi.LocalTarget(
        Path() / "data" / "test_messages.csv"
      ),
    }
