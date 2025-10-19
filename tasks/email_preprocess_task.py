"""Preprocessing of retrieved email spam data."""

from email import message_from_binary_file
from email.charset import Charset
from email.message import EmailMessage
from email.policy import default
from pathlib import Path
from tarfile import open as tar_open, TarFile
from typing import override

import bs4
import luigi
import pandas as pd

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
        message = None
        try:
          message = _parse_data(email_message)
        except (LookupError, UnicodeDecodeError) as e:
          print(e)
        if message is not None:
          spam_data["message"] += [message]
          spam_data["type"] += ["email"]
          spam_data["is_spam"] += [int(is_spam)]
        file.close()


def _parse_data(message: EmailMessage) -> str | None:
  content = []
  has_alternative_parts = False
  part: EmailMessage
  for part in message.walk():
    content_type = part.get_content_type()
    if content_type == "multipart/alternative":
      has_alternative_parts = True
      continue
    elif part.get_content_maintype() == "multipart":
      continue
    elif content_type == "text/plain":
      payload = _parse_payload(part)
      content += [payload]
    elif content_type == "text/html":
      payload = _parse_payload(part)
      content += [bs4.BeautifulSoup(payload, "html.parser").get_text()]
    else:
      pass
    if has_alternative_parts:
      break
  return "".join(content) if content else None


def _parse_payload(message: EmailMessage) -> str:
  charset_str = message.get_content_charset()
  if charset_str is not None:
    charset = Charset(charset_str)
    if charset.input_codec is not None:
      return str(
        message.get_payload(decode=True).decode(charset.input_codec)  # type: ignore
      )
  return str(
    message.get_payload(decode=True).decode("ascii", errors="replace")  # type: ignore
  )


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
