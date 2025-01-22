"""Utilities for accessing Scikit-learn pipelines."""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from common.types import PipelineStep


def get_transformers(model: Pipeline) -> list[BaseEstimator]:
  return (model[:-1]
          if isinstance(model, Pipeline)
          else model)


def get_predictor_name(model: Pipeline) -> str:
  return (model.steps[-1][0]
          if isinstance(model, Pipeline)
          else model.__class__.__name__)


def get_predictor(model: Pipeline) -> BaseEstimator:
  return (model.steps[-1][1]
          if isinstance(model, Pipeline)
          else model)


def get_predictor_data(model: Pipeline) -> PipelineStep:
  return get_predictor_name(model), get_predictor(model)
