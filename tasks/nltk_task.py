"""Access to NLTK corpora.

Corpora are downloaded for names, stopwords and words.
"""

import os
from pathlib import Path
from typing import override

import luigi
import nltk
import nltk.corpus


class NltkTask(luigi.Task):
  """Downloads NLTK corpora for names, words and stopwords.

  There is no language limitation, the intention is to support
  multiple languages even if the messages mostly contain English text.

  The words in all the corpora are additionally converted to lowercase,
  in order to be comparable to tokenized data.

  Since most of the message content is in English, an instance of
  `nltk.PorterStemmer` is used for stemming. For the purposes of
  English language detection, all English words are stored in a separate set.
  """

  @override
  def run(self):
    nltk.download("names")
    nltk.download("words")
    nltk.download("stopwords")

  @override
  def output(self):
    env_home = os.environ["HOME"]
    return [
      luigi.LocalTarget(
        Path() / env_home / "nltk_data" / "corpora" / "names.zip"
      ),
      luigi.LocalTarget(
        Path() / env_home / "nltk_data" / "corpora" / "words.zip"
      ),
      luigi.LocalTarget(
        Path() / env_home / "nltk_data" / "corpora" / "stopwords.zip"
      ),
    ]

  @property
  def all_names(self) -> set[str]:
    if not hasattr(self, "_all_names"):
      self._all_names = set([
        word.lower() for word in nltk.corpus.names.words()
      ])
    return self._all_names

  @property
  def all_english_words(self) -> set[str]:
    if not hasattr(self, "_all_english_words"):
      self._all_english_words = set([
        word.lower()
        for word in nltk.corpus.words.words(["en", "en-basic"])
      ])
    return self._all_english_words

  @property
  def all_stopwords(self) -> set[str]:
    if not hasattr(self, "_all_stopwords"):
      self._all_stopwords = set([
        word.lower() for word in nltk.corpus.stopwords.words()
      ])
    return self._all_stopwords

  @property
  def stemmer(self) -> nltk.PorterStemmer:
    if not hasattr(self, "_stemmer"):
      self._stemmer = nltk.PorterStemmer()
    return self._stemmer
