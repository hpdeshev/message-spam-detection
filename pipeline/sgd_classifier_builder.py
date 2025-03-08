"""Trains bag-of-words (BoW) models via stochastic gradient descent (SGD)."""

from collections.abc import Collection
import copy

import numpy as np
from sklearn.metrics import accuracy_score, hinge_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from typing_extensions import override

from common.config import misc
from pipeline.text_classifier_builder import TextClassifierBuilder
from pipeline.utils import get_predictor, get_transformers
from tasks.utils import get_sample_weights


class SgdClassifierBuilder(TextClassifierBuilder):
  """Specialized builder of `SGD` `BoW` classifiers.
  
  Executes the predictor's `partial_fit` method on per-epoch basis.
  A validation set and early stopping are used for additional
  regularization and to prevent overfitting.
  """

  @override
  def _fit(
    self,
    model: Pipeline,
    X: Collection[str],
    y: Collection[int],
    balanced_weights: bool = True,
    incremental: bool = False,
    verbose: bool = False,
    params: dict[str, bool | float | int | str] | None = None,
  ) -> Pipeline:
    transformers = get_transformers(model)
    if incremental:
      X_tfidf = transformers.transform(X)
    else:
      X_tfidf = transformers.fit_transform(X, y)

    predictor = get_predictor(model)
    params = params if params is not None else {}
    predictor.set_params(**params)

    predictor_params = predictor.get_params()
    max_iter = predictor_params["max_iter"]
    predictor.set_params(max_iter=1, early_stopping=False)

    (X_train, X_val,
     y_train, y_val) = train_test_split(
      X_tfidf, y,
      test_size=predictor_params["validation_fraction"],
      random_state=misc().random_seed, shuffle=True,
      stratify=y,
    )
    sample_weights = (
      get_sample_weights(y_train) if balanced_weights else None
    )
    best_val_loss, last_good_model = None, None
    n_iter, n_iter_no_change, is_looping = 0, 0, True
    while is_looping and (n_iter < max_iter):
      predictor.partial_fit(
        X_train, y_train,
        classes=np.unique(y_train),
        sample_weight=sample_weights,
      )
      if predictor_params["loss"] == "hinge":
        val_loss = hinge_loss(
          y_val, predictor.decision_function(X_val)
        )
      else:
        val_loss = log_loss(
          y_val, predictor.predict_proba(X_val)[:, 1]
        )
      if best_val_loss is None:
        best_val_loss = val_loss
        last_good_model = copy.deepcopy(model)
      else:
        if best_val_loss - predictor_params["tol"] < val_loss:
          n_iter_no_change += 1
          print(f"{n_iter_no_change} iteration(s) without improvement.")
          if n_iter_no_change == predictor_params["n_iter_no_change"]:
            print(f"Reached {predictor_params['n_iter_no_change']} "
                  "iteration(s) without improvement.")
            is_looping = False
        else:
          n_iter_no_change = 0
          best_val_loss = val_loss
          last_good_model = copy.deepcopy(model)
      if verbose:
        train_acc = accuracy_score(
          y_train, predictor.predict(X_train)
        )
        val_acc = accuracy_score(
          y_val, predictor.predict(X_val)
        )
        print(f"train_accuracy={train_acc:.3f}, "
              f"val_accuracy={val_acc:.3f}, "
              f"val_loss={val_loss:.3f}")
      n_iter += 1
    return last_good_model
