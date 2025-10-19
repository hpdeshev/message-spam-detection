"""Preprocessing of retrieved SMS spam data."""

from pathlib import Path
import re
from typing import override
from zipfile import ZipFile

import luigi
import pandas as pd

from common.types import SpamDict
from tasks.message_retrieval_task import MessageRetrievalTask


_FILES = ["sms+spam+collection.zip"]
_URL = "https://archive.ics.uci.edu/static/public/228/"


def _parse_dataset_from_zipfile(
  zfile: ZipFile,
  spam_data: SpamDict,
) -> None:
  for zipobj in zfile.infolist():
    if not zipobj.filename.endswith("readme"):
      with zfile.open(zipobj.filename, mode="r") as file:
        for line in file:
          message, is_spam = _parse_data(line)
          spam_data["message"] += [message]
          spam_data["type"] += ["sms"]
          spam_data["is_spam"] += [is_spam]


def _parse_data(payload: bytes) -> tuple[str, int]:
  message_tag, message = re.split(
    r"\s+", payload.decode().strip(), maxsplit=1
  )
  return message, int(message_tag == "spam")


class SmsPreprocessTask(luigi.Task):
  """Outputs a preprocessed SMS dataset.

  SMS messages and their labels are extracted from a `zip` archive,
  in which a `readme` file is skipped as it does not contain SMS data.
  """

  @override
  def requires(self):
    return MessageRetrievalTask(
      _FILES, "data", _URL
    )

  @override
  def run(self):
    spam_data: dict[str, list[str | int]] = {
      "message": [], "type": [], "is_spam": []
    }
    for file in _FILES:
      with ZipFile(Path() / "data" / file) as zfile:
        _parse_dataset_from_zipfile(zfile, spam_data)
    spam_df = pd.DataFrame(spam_data)
    spam_df.to_csv(self.output().path, index=False)

  @override
  def output(self):
    return luigi.LocalTarget(
      Path() / "data" / "sms_spam_data.csv"
    )
