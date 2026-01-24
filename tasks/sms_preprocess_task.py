"""Preprocessing of retrieved SMS spam data."""

from pathlib import Path
import re
from typing import override
from zipfile import ZipFile

import luigi
import pandas as pd

from common.types import SpamData
from tasks.message_retrieval_task import MessageRetrievalTask


_FILES = ["sms+spam+collection.zip"]
_URL = "https://archive.ics.uci.edu/static/public/228/"
_DATA_PATH = Path("data")
_OUTPUT_PATH = _DATA_PATH / "sms_spam_data.csv"


def _parse_dataset_from_zipfile(
  zfile: ZipFile,
  spam_data: SpamData,
) -> None:
  for zipobj in zfile.infolist():
    if not zipobj.filename.endswith("readme"):
      with zfile.open(zipobj.filename, mode="r") as file:
        for line in file:
          message, is_spam = _parse_data(line)
          spam_data["message"].append(message)
          spam_data["kind"].append("sms")
          spam_data["is_spam"].append(is_spam)


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
    spam_data = SpamData(message=[], kind=[], is_spam=[])
    for file in _FILES:
      with ZipFile(_DATA_PATH / file) as zfile:
        _parse_dataset_from_zipfile(zfile, spam_data)
    spam_df = pd.DataFrame(spam_data)
    spam_df.to_csv(_OUTPUT_PATH, index=False)

  @override
  def output(self):
    return luigi.LocalTarget(_OUTPUT_PATH)
