"""General configuration."""

import luigi


class misc(luigi.Config):
  random_seed = luigi.IntParameter(42)


class classification(luigi.Config):
  feature_selector_type = luigi.Parameter("model")
  validation_split = luigi.FloatParameter(0.1)
