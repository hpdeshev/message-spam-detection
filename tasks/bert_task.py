"""A bidirectional encoder representations from transformers (BERT) builder."""

from collections.abc import Iterable, MutableMapping
from pathlib import Path
import re
from typing import Any, cast, override

import datasets
import evaluate
import luigi
import numpy as np
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

from common.config import classification, misc, tokenization
from pipeline.text_classifier_builder import TextClassifierBuilder
from pipeline.utils import get_transformers
from tasks.best_bow_task import BestBowTask
from tasks.train_test_split_task import TrainTestSplitTask


_CLASSIFIER_PATH = Path("models") / "bert_classifier"
_OUTPUT_MODEL_PATH = _CLASSIFIER_PATH / "model"
_OUTPUT_VOCAB_PATH = _CLASSIFIER_PATH / "vocab.txt"
_REGEX_SEPARATORS = tokenization().regex_separators


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
  bow_vocabulary: set[str] = set()
  regular_prefix, custom_prefix = "TF-IDF__", "Custom TF-IDF__"
  feature_names = get_transformers(bow_classifier).get_feature_names_out()
  for name in feature_names:
    if name.startswith(regular_prefix):
      bow_vocabulary = bow_vocabulary.union(
        name.removeprefix(regular_prefix).split()
      )
    elif name.startswith(custom_prefix):
      bow_vocabulary = bow_vocabulary.union(
        name.removeprefix(custom_prefix).split()
      )
    else:
      raise ValueError("Unexpected feature name prefix.")
  return bow_vocabulary


def get_bow_dataset(
  ds: datasets.Dataset,
  bow_classifier: Pipeline,
  bow_vocabulary: set[str] | None = None,
) -> datasets.Dataset:
  """Converts `get_bow_vocabulary`'s output to a message dataset.

  Together with `get_bow_vocabulary`, this function makes it possible for
  `BERT` and `BoW` classifiers to learn from the same dataset and
  thus their results on the dataset can be compared.

  Args:
    ds: A `datasets.Dataset` instance which contains messages and their
      ham/spam labels.
    bow_classifier: A `Scikit-learn` pipeline.
    bow_vocabulary: Optional, a set of `BoW` vocabulary items.
      Can be initialized internally via the `get_bow_vocabulary` function.

  Returns:
    A `datasets.Dataset` instance.
  """
  if bow_vocabulary is None:
    bow_vocabulary = get_bow_vocabulary(bow_classifier)
  features_step = bow_classifier.named_steps["Features"]
  tfidf_features_step = features_step["TF-IDF"]
  custom_tfidf_features_step = features_step["Custom TF-IDF"]
  custom_feature_extractor = (
    custom_tfidf_features_step.named_steps["customfeatureextractor"]
  )
  feature_detectors = [
    detector
    for detector in custom_feature_extractor.detectors
    if detector.feature_name in bow_vocabulary
  ]
  def preprocess_message(
    row: MutableMapping[str, Any]
  ) -> MutableMapping[str, Any]:
    tokens = []
    for token in re.split(_REGEX_SEPARATORS, row["message"]):
      if token:
        bow_token = tfidf_features_step.tokenizer(token)
        bow_token = bow_token[0] if bow_token else token
        if bow_token in bow_vocabulary:
          tokens.append(bow_token)
        else:
          for detector in feature_detectors:
            if detector.check(bow_token):
              tokens.append(detector.feature_name)
    row["message"] = " ".join(tokens)
    return row
  return ds.map(preprocess_message)


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

  max_input_tokens = luigi.IntParameter(default=128)
  model_name = luigi.Parameter(default="google/bert_uncased_L-2_H-128_A-2")

  @override
  def requires(self):
    return {
      "train_test_split": TrainTestSplitTask(),
      "vocab_provider": BestBowTask(),
    }

  @override
  def run(self):
    bow_builder = TextClassifierBuilder()
    bow_classifier = bow_builder.build(
      self.input()["vocab_provider"]["best_bow_classifier"].path
    )
    bow_vocabulary = get_bow_vocabulary(bow_classifier)

    vocab_filepath = _OUTPUT_VOCAB_PATH
    vocab_filepath.parent.mkdir(parents=True, exist_ok=True)
    vocab_filepath.write_text("\n".join(bow_vocabulary))

    dataset = datasets.Dataset.from_csv(
      self.input()["train_test_split"]["train"].path
    )
    dataset = dataset.remove_columns("kind")
    dataset = get_bow_dataset(dataset, bow_classifier, bow_vocabulary)
    dataset = dataset.class_encode_column("is_spam")
    dataset = dataset.train_test_split(
      test_size=classification().validation_split,
      seed=misc().random_seed, shuffle=True,
      stratify_by_column="is_spam",
    )
    tokenizer = BertTokenizer(
      vocab_filepath,
      model_max_length=self.max_input_tokens,
      do_basic_tokenize=False,
    )
    def preprocess(data: dict[str, Any]) -> dict[str, Any]:
      return {"input_ids":
                tokenizer(data["message"], truncation=True)["input_ids"]}
    dataset = dataset.map(preprocess, batched=True, remove_columns="message")
    dataset = dataset.rename_column("is_spam", "labels")
    train_ds, val_ds = dataset["train"], dataset["test"]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = BertForSequenceClassification(
      AutoConfig.from_pretrained(self.model_name)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    training_args = TrainingArguments(
      output_dir=str(_OUTPUT_MODEL_PATH),
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      weight_decay=0.01,
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=1,
      seed=misc().random_seed,
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
        raise EnvironmentError("Accuracy module not run on the main process.")
      return result
    y = train_ds["labels"]
    class_weights = compute_class_weight(
      class_weight="balanced",
      classes=np.unique(y),
      y=y,
    )
    class CustomTrainer(Trainer):
      def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
      ) -> tuple[torch.Tensor, SequenceClassifierOutput] | torch.Tensor:
        model = cast(BertForSequenceClassification, model)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
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
  def output(self):
    return {
      "bert_classifier_model":
        luigi.LocalTarget(_OUTPUT_MODEL_PATH),
      "bert_classifier_vocab":
        luigi.LocalTarget(_OUTPUT_VOCAB_PATH),
    }
