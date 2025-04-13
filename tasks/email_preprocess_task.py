"""Preprocessing of retrieved email spam data."""

from pathlib import Path
from tarfile import open as tar_open, TarFile

import bs4
from email.message import Message
from email.parser import BytesParser
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
  parser: BytesParser,
  tar: TarFile,
  is_spam: bool,
  spam_data: SpamDict,
) -> None:
  for tarobj in tar.getmembers()[1:]:
    if not tarobj.name.endswith("cmds"):
      file = tar.extractfile(tarobj)
      if file is not None:
        message = parser.parse(file)  # type: ignore
        content = _parse_data(
          _parse_payload(message),
          message.get_content_type(),
        )
        spam_data["message"] += [content.replace("\x00", "")]
        spam_data["type"] += ["email"]
        spam_data["is_spam"] += [int(is_spam)]
        file.close()


def _parse_data(
  payload: list[Message] | str,
  content_type: str,
) -> str:
  if isinstance(payload, list):
    content = ""
    for message in payload:
      if (
        message.get_content_disposition() != "attachment"
        and message.get_content_maintype() in ["message", "multipart", "text"]
      ):
        content += _parse_data(
          _parse_payload(message),
          message.get_content_type(),
        )
    return content

  if content_type == "text/html":
    payload = bs4.BeautifulSoup(payload, "html.parser").get_text()
  return payload


def _parse_payload(email: Message) -> list[Message] | str:
  payload = None
  if email.is_multipart():
    payload = email.get_payload(decode=False)
  else:
    try:
      payload = email.get_payload(decode=True)
      payload = payload.decode("unicode_escape")  # type: ignore
    except Exception as e:
      print("parse_payload:", e, sep="\n")
      payload = email.get_payload(decode=False)
  return payload  # type: ignore


class EmailPreprocessTask(luigi.Task):
  """Ouputs a preprocessed email dataset.

  Email spam data is extracted from `tar.bz2` archives,
  in which `cmds` files are skipped as they do not contain email data.

  For the purposes of text classification, in emails only plain or HTML text
  payload is processed and email headers and attachments are ignored.
  """

  @override
  def requires(self):  # type: ignore
    return MessageRetrievalTask(
      _ALL_FILES, "data", _URL
    )

  @override
  def run(self):
    parser = BytesParser()
    spam_data = {"message": [], "type": [], "is_spam": []}
    for file in _ALL_FILES:
      is_spam = file in _SPAM_FILES
      with tar_open(Path() / "data" / file, mode="r:bz2") as tar:
        _parse_dataset_from_tarfile(
          parser, tar, is_spam, spam_data
        )
    spam_df = pd.DataFrame(spam_data)
    spam_df.to_csv(self.output().path)

  @override
  def output(self):  # type: ignore
    return luigi.LocalTarget(
      Path() / "data" / "email_spam_data.csv"
    )
