"""Preprocessing of retrieved email spam data."""

from email import message_from_binary_file
from email.message import EmailMessage
from email.policy import default
from pathlib import Path
from tarfile import open as tar_open, TarFile

import bs4
import luigi
import pandas as pd
from typing_extensions import override

from common.types import SpamDict
from tasks.message_retrieval_task import MessageRetrievalTask


_HAM_FILES = [
  "20030228_easy_ham.tar.bz2",
  "20030228_easy_ham_2.tar.bz2",
  "20030228_hard_ham.tar.bz2",
]
_SPAM_FILES = [
  "20030228_spam.tar.bz2",
  "20050311_spam_2.tar.bz2",
]
_ALL_FILES = _HAM_FILES + _SPAM_FILES
_URL = "https://spamassassin.apache.org/old/publiccorpus/"


def _parse_dataset_from_tarfile(
  tar: TarFile,
  is_spam: bool,
  spam_data: SpamDict,
) -> None:
  for tarobj in tar.getmembers()[1:]:
    if not tarobj.name.endswith("cmds"):
      file = tar.extractfile(tarobj)
      if file is not None:
        email_message = message_from_binary_file(file, policy=default)
        message = _parse_data(email_message)
        if message:
          spam_data["message"] += [message]
          spam_data["type"] += ["email"]
          spam_data["is_spam"] += [int(is_spam)]
        file.close()


def _parse_data(message: EmailMessage) -> str:
  content = []
  for part in message.walk():
    if part.get_content_maintype() == "multipart":
        continue
    content_type = part.get_content_type()
    if content_type == "text/plain":
      payload = str(part.get_payload())
      content += [payload]
    elif content_type == "text/html":
      payload = str(part.get_payload())
      content += [bs4.BeautifulSoup(payload, "html.parser").get_text()]
    else:
      pass
  return "".join(content)


class EmailPreprocessTask(luigi.Task):
  """Ouputs a preprocessed email dataset.

  Email spam data is extracted from `tar.bz2` archives,
  in which `cmds` files are skipped as they do not contain email data.

  For the purposes of text classification, in emails only plain or HTML text
  payload is processed and email headers and attachments are ignored.
  """

  @override
  def requires(self):
    return MessageRetrievalTask(
      _ALL_FILES, "data", _URL
    )

  @override
  def run(self):
    spam_data: dict[str, list[str | int]] = {
      "message": [], "type": [], "is_spam": []
    }
    for file in _ALL_FILES:
      is_spam = file in _SPAM_FILES
      with tar_open(Path() / "data" / file, mode="r:bz2") as tar:
        _parse_dataset_from_tarfile(
          tar, is_spam, spam_data
        )
    spam_df = pd.DataFrame(spam_data)
    spam_df.to_csv(self.output().path, index=False)

  @override
  def output(self):
    return luigi.LocalTarget(
      Path() / "data" / "email_spam_data.csv"
    )
