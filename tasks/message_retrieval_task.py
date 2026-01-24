"""Implementation of message data retrieval."""

import io
import os
from pathlib import Path
from typing import override

import luigi
import luigi.format
import requests


class MessageRetrievalTask(luigi.Task):
  """Downloads messages from a URL to the local file system."""

  message_files = luigi.ListParameter()
  message_folder = luigi.Parameter()
  message_url = luigi.Parameter()

  @override
  def run(self):
    for filepath in self.output():
      if not filepath.exists():
        response = requests.get(
          self.message_url + os.path.basename(filepath.path)
        )
        fileobj = io.BytesIO(response.content)
        with filepath.open("w") as f:
          f.write(fileobj.getbuffer())

  @override
  def output(self):
    message_folder_path = Path(self.message_folder)
    return [
      luigi.LocalTarget(
        message_folder_path / file,
        format=luigi.format.Nop,
      )
      for file in self.message_files
    ]
