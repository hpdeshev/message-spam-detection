"""General configuration."""

import luigi


class misc(luigi.Config):
  random_seed = luigi.IntParameter(default=42)


class tokenization(luigi.Config):
  currencies = luigi.Parameter(default="$£€")
  regex_separators = luigi.Parameter(default=r"[^\w$£€]+")


class classification(luigi.Config):
  feature_selector_type = luigi.Parameter(default="model")
  validation_split = luigi.FloatParameter(default=0.1)
