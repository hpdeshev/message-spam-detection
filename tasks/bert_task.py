"""A bidirectional encoder representations from transformers (BERT) builder."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import datasets
import luigi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.keras.models as tf_models
import transformers
from typing_extensions import override

from common.config import classification, misc
from common.types import TokenizerMethod
from pipeline.text_classifier_builder import TextClassifierBuilder
from pipeline.utils import get_transformers
from tasks.best_bow_task import BestBowTask
from tasks.train_test_split_task import TrainTestSplitTask


def get_bow_vocabulary(bow_classifier: Pipeline) -> set[str]:
  """Returns a `BoW` classifier's learned vocabulary as a set of tokens.

  Together with `get_bow_dataset`, this function makes it possible for
  `BERT` and `BoW` classifiers to learn from the same dataset and
  thus their results on the dataset can be compared.
  """
  bow_vocabulary = set()
  feature_names = [name.split("TF-IDF__")[1]
                   for name in (get_transformers(bow_classifier)
                                .get_feature_names_out())
                   if name.startswith("TF-IDF__")]
  for name in feature_names:
    ngrams = name.split()
    if len(ngrams) > 1:
      bow_vocabulary = bow_vocabulary.union(ngrams)
    else:
      bow_vocabulary.add(ngrams[0])
  return bow_vocabulary


def get_bow_dataset(
  bow_vocabulary: set[str],
  bow_tokenizer: TokenizerMethod,
  X: Iterable[str],
  y: Iterable[int]
) -> pd.DataFrame:
  """Converts `get_bow_vocabulary`'s output to a message dataset.
  
  Together with `get_bow_vocabulary`, this function makes it possible for
  `BERT` and `BoW` classifiers to learn from the same dataset and
  thus their results on the dataset can be compared. 
  """
  dataset = {"message" : [], "is_spam" : []}
  for message, is_spam in zip(X, y):
    dataset["message"].append(
      " ".join([
        token
        for token in bow_tokenizer(message)
        if token in bow_vocabulary
      ])
    )
    dataset["is_spam"].append(is_spam)
  return pd.DataFrame(dataset)


class BertTask(luigi.Task):
  """Outputs a `BERT` classifier.
  
  Implements an end-to-end classification task with the following steps:

  1. A `BoW` dataset is obtained via the `get_bow_dataset` function;
  2. The `BoW` dataset is split into subsets for training and test;
  3. A `BERT` `Keras` model is loaded;
  4. The `BERT` model is compiled with learning rate scheduling and
     an optimizer;
  5. A `BERT` tokenizer wrapper tokenizes and dynamically pads the
     training and test data in preparation for training and evaluation;
  6. The `BERT` model is trained with class weighting, validation set and
     early stopping;
  7. The `BERT` model is saved.
  """

  max_input_tokens = luigi.IntParameter(128)
  model_name = luigi.Parameter("google/bert_uncased_L-2_H-128_A-2")

  @override
  def requires(self):
    return {
      "train_test_split" : TrainTestSplitTask(),
      "vocab_provider" : BestBowTask(),
    }

  @override
  def run(self):
    bow_builder = TextClassifierBuilder()
    bow_classifier = bow_builder.build(
      self.input()["vocab_provider"]["best_bow_classifier"].path
    )
    bow_vocabulary = get_bow_vocabulary(bow_classifier)
    train_df = pd.read_csv(
      self.input()["train_test_split"]["train"].path, index_col=0
    )
    bert_train_df = get_bow_dataset(
      bow_vocabulary, bow_classifier["Features"]["TF-IDF"].tokenizer,
      train_df.message, train_df.is_spam
    )

    (X_train, X_val,
     y_train, y_val) = train_test_split(
      bert_train_df.message, bert_train_df.is_spam,
      test_size=classification().validation_split,
      random_state=misc().random_seed, shuffle=True,
      stratify=bert_train_df.is_spam
    )

    batch_size = 32
    max_epochs = 20
    max_epochs_no_change = 3
    batches_per_epoch = len(y_train) // batch_size
    total_train_steps = int(batches_per_epoch * max_epochs)
    optimizer, _ = transformers.create_optimizer(
      init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps
    )

    classifier = \
      transformers.TFAutoModelForSequenceClassification.from_pretrained(
        self.model_name,
        num_labels=2,
        id2label={ 0 : "Ham", 1 : "Spam" },
        label2id={ "Ham" : 0, "Spam" : 1 },
      )
    classifier.build()
    classifier.summary()
    classifier.compile(optimizer=optimizer)
    preprocessor = BertPreprocessor(self.model_name, self.max_input_tokens)

    train_set = datasets.Dataset.from_pandas(
      pd.concat([X_train, y_train], axis=1)
    )
    tokenized_train_set = preprocessor.to_tf_dataset(
      classifier, train_set, shuffle=True
    )

    val_set = datasets.Dataset.from_pandas(
      pd.concat([X_val, y_val], axis=1)
    )
    tokenized_val_set = preprocessor.to_tf_dataset(
      classifier, val_set
    )

    best_val_loss = None
    n_epoch, n_epoch_no_change, is_looping = 0, 0, True
    while is_looping and (n_epoch < max_epochs):
      history = classifier.fit(
        tokenized_train_set,
        validation_data=tokenized_val_set,
        class_weight=dict(enumerate(
          len(y_train) / (2 * np.bincount(y_train))
        ))
      )
      val_loss = history.history["val_loss"][-1]
      if best_val_loss is None:
        best_val_loss = val_loss
      else:
        if best_val_loss < val_loss:
          n_epoch_no_change += 1
          print(f"{n_epoch_no_change} epoch(s) without improvement.")
          if n_epoch_no_change == max_epochs_no_change:
            print(f"Reached {n_epoch_no_change} epoch(s) "
                  "without improvement.")
            is_looping = False
        else:
          n_epoch_no_change = 0
          best_val_loss = val_loss
    classifier.save_pretrained(self.output().path)

  @override
  def output(self):
    return luigi.LocalTarget(
      Path() / "models" / "bert_classifier"
    )


class BertPreprocessor:
  """Wrapper for a `BERT` preprocessor from Hugging Face Transformers.
  
  Data collation for dynamic batch-based padding is used.
  """

  def __init__(
    self,
    model_name: str,
    max_input_tokens: int
  ):
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name, model_max_length=max_input_tokens
    )
    self._data_collator = transformers.DataCollatorWithPadding(
      tokenizer=self._tokenizer, return_tensors="tf"
    )

  @property
  def tokenizer(self):
    return self._tokenizer

  def to_tf_dataset(
    self,
    bert_classifier: tf_models.Model,
    hf_dataset: datasets.Dataset,
    batched: bool = True,
    batch_size: int = 32,
    shuffle: bool = False
  ) -> tf_data.Dataset:
    tokenized_dataset = hf_dataset.map(self._preprocess, batched=batched)
    tokenized_dataset = bert_classifier.prepare_tf_dataset(
      tokenized_dataset,
      shuffle=shuffle,
      batch_size=batch_size,
      collate_fn=self._data_collator
    )
    return tokenized_dataset

  def _preprocess(
    self,
    data: Mapping[str, list[str | int]]
  ) -> dict[str, list[list[int] | int]]:
    result = {
      "input_ids" : self._tokenizer(
                      data["message"], truncation=True
                    )["input_ids"],
    }
    if "is_spam" in data:
      result.update({
        "labels" : data["is_spam"]
      })
    return result
