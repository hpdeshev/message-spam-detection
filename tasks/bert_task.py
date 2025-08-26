"""A bidirectional encoder representations from transformers (BERT) builder."""

from collections.abc import Iterable, Mapping
from pathlib import Path
import re

import datasets
import evaluate
import luigi
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from transformers import (
  AutoConfig, DataCollatorWithPadding,
  EarlyStoppingCallback,
  TrainingArguments, Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import (
  BertForSequenceClassification, BertTokenizer
)
from transformers.trainer_utils import EvalPrediction
from typing_extensions import Any, override

from common.config import classification, misc, tokenization
from pipeline.text_classifier_builder import (
  FEATURE_DISCOVERY_METHODS, TextClassifierBuilder
)
from pipeline.utils import get_transformers
from tasks.best_bow_task import BestBowTask
from tasks.train_test_split_task import TrainTestSplitTask


def get_bow_vocabulary(bow_classifier: Pipeline) -> set[str]:
  """Returns a `BoW` classifier's learned vocabulary as a set of tokens.

  Together with `get_bow_dataset`, this function makes it possible for
  `BERT` and `BoW` classifiers to learn from the same dataset and
  thus their results on the dataset can be compared.

  Args:
    bow_classifier: A `Scikit-learn` pipeline.

  Returns:
    A set of `BoW` vocabulary items.
  """
  bow_vocabulary = set()
  feature_names = [name.removeprefix("TF-IDF__")
                   for name in (get_transformers(bow_classifier)
                                .get_feature_names_out())
                   if name.startswith("TF-IDF__")]
  custom_feature_names = [
    method[0] for method in (bow_classifier
                             .named_steps["Features"]["Custom TF-IDF"]
                             .named_steps["customfeatureextractor"].methods)
  ]
  feature_names += custom_feature_names
  for name in feature_names:
    ngrams = name.split()
    if len(ngrams) > 1:
      bow_vocabulary = bow_vocabulary.union(ngrams)
    else:
      bow_vocabulary.add(ngrams[0])
  return bow_vocabulary


def get_bow_dataset(
  bow_classifier: Pipeline,
  bow_vocabulary: set[str],
  X: Iterable[str],
  y: Iterable[int],
) -> pd.DataFrame:
  """Converts `get_bow_vocabulary`'s output to a message dataset.
  
  Together with `get_bow_vocabulary`, this function makes it possible for
  `BERT` and `BoW` classifiers to learn from the same dataset and
  thus their results on the dataset can be compared.

  Args:
    bow_classifier: A `Scikit-learn` pipeline.
    bow_vocabulary: A set of `BoW` vocabulary items.
    X: The messages.
    y: The ham/spam labels.

  Returns:
    A `Pandas` data frame.
  """
  dataset = {"message": [], "is_spam": []}
  feature_discovery_methods = [
    method
    for method in (bow_classifier
                   .named_steps["Features"]["Custom TF-IDF"]
                   .named_steps["customfeatureextractor"].methods)
  ]
  for message, is_spam in zip(X, y):
    tokens = []
    for token in re.split(tokenization().regex_separators,
                          message.lower()):
      if token:
        bow_token = (bow_classifier
                     .named_steps["Features"]["TF-IDF"]
                     .tokenizer(token))
        bow_token = bow_token[0] if bow_token else token
        if bow_token in bow_vocabulary:
          tokens += [bow_token]
        for feature, checker in feature_discovery_methods:
          if checker([bow_token]):
            tokens += [feature]
    dataset["message"] += [" ".join(tokens)]
    dataset["is_spam"] += [is_spam]
  return pd.DataFrame(dataset)


class BertTask(luigi.Task):
  """Outputs a `BERT` classifier.
  
  Implements an end-to-end classification task with the following steps:

  1. A `BoW` dataset is obtained via the `get_bow_dataset` function;
  2. The `BoW` dataset is split into subsets for training and test;
  3. A `BERT` `PyTorch` model is loaded;
  4. A `BERT` tokenizer tokenizes and dynamically pads the
     training and test data in preparation for training and evaluation;
  5. The `BERT` model is trained with class weighting, validation set and
     early stopping;
  6. The `BERT` model is persisted.

  To support straightforward comparison between `BERT` and `BoW` classifiers:
  - `WordPiece` tokenization is not used and the saved at `vocab_filepath`
    output from the `get_bow_vocabulary` function is directly used instead;
  - a *not* pretrained model with `BERT` architecture is used, meaning that
    the embedding matrix is as per the `BoW` vocabulary, i.e., the matrix may
    not be used entirely and the token indices have different meaning compared
    to the original `BERT`.
  """

  max_input_tokens = luigi.IntParameter(128)
  model_name = luigi.Parameter("google/bert_uncased_L-2_H-128_A-2")

  @override
  def requires(self):  # type: ignore
    return {
      "train_test_split": TrainTestSplitTask(),
      "vocab_provider": BestBowTask(),
    }

  @override
  def run(self):
    bow_builder = TextClassifierBuilder()
    bow_classifier = bow_builder.build(
      self.input()["vocab_provider"]["best_bow_classifier"].path  # type: ignore
    )
    bow_vocabulary = get_bow_vocabulary(bow_classifier)

    vocab_filepath = Path(self.output()["bert_classifier_vocab"].path)
    vocab_filepath.parent.mkdir(parents=True, exist_ok=True)
    vocab_filepath.write_text("\n".join([item for item in bow_vocabulary]))

    train_df = pd.read_csv(
      self.input()["train_test_split"]["train"].path  # type: ignore
    )
    bert_train_df = get_bow_dataset(
      bow_classifier, bow_vocabulary,
      train_df.message, train_df.is_spam,
    )

    tokenizer = BertTokenizer(
      vocab_filepath,
      model_max_length=self.max_input_tokens,
      do_basic_tokenize=False,
    )
    dataset = datasets.Dataset.from_pandas(bert_train_df)
    dataset = dataset.class_encode_column("is_spam")
    dataset = dataset.train_test_split(
      test_size=classification().validation_split,  # type: ignore
      seed=misc().random_seed, shuffle=True,  # type: ignore
      stratify_by_column="is_spam",
    )
    def preprocess(
      data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
      return {
        "input_ids": tokenizer(
                      data["message"], truncation=True
                     )["input_ids"],
        "labels": data["is_spam"],
      }
    train_ds = dataset["train"].map(preprocess, batched=True)
    val_ds = dataset["test"].map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = BertForSequenceClassification(
      AutoConfig.from_pretrained(self.model_name)  # type: ignore
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # type: ignore
    training_args = TrainingArguments(
      output_dir=self.output()["bert_classifier_model"].path,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      weight_decay=0.01,
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=1,
      seed=misc().random_seed,  # type: ignore
      load_best_model_at_end=True,
      report_to=["tensorboard"],
      push_to_hub=False,
      num_train_epochs=100,
    )
    accuracy = evaluate.load("accuracy")
    def compute_metrics(
      eval_pred: EvalPrediction
    ) -> dict[str, float]:
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      result = accuracy.compute(predictions=predictions, references=labels)
      if result is None:
        raise EnvironmentError(
          "Accuracy module not run on the main process."
        )
      return result
    classes = np.unique(train_df.is_spam)
    class CustomTrainer(Trainer):
      def compute_loss(
        self,
        model: BertForSequenceClassification,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
      ) -> tuple[torch.Tensor, SequenceClassifierOutput] | torch.Tensor:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        class_weights = compute_class_weight(
          class_weight="balanced",
          classes=classes,
          y=train_df.is_spam,
        )
        loss_fct = nn.CrossEntropyLoss(
          weight=torch.tensor(
            class_weights, device=model.device, dtype=torch.float32
          )
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels),
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    trainer.save_model()
    print(trainer.evaluate())

  @override
  def output(self):  # type: ignore
    return {
      "bert_classifier_model":
        luigi.LocalTarget(Path() / "models" / "bert_classifier" / "model"),
      "bert_classifier_vocab":
        luigi.LocalTarget(Path() / "models" / "bert_classifier" / "vocab.txt"),
    }
