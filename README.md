# Demonstration of methods for message spam detection
## Overview
This repository demonstrates and compares various message spam detection methods by implementing a general ham/spam binary classification task on datasets obtained from:
- [SpamAssassin public mail corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [SMS Spam Collection - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The ham/spam binary classification task is implemented by utilization of the following two groups of machine learning models:
- [bag-of-words (BoW)](https://en.wikipedia.org/wiki/Bag-of-words_model)
  - [naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
  - [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
  - [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
  - [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine)
  - [ensembles - bagging (voting), boosting, stacking](https://en.wikipedia.org/wiki/Ensemble_learning)
- [bidirectional encoder representations from transformers (BERT)](https://en.wikipedia.org/wiki/BERT_(language_model))

The message spam data is modeled in two different ways depending on the classification approach:
- [term frequency–inverse document frequency (`TF-IDF`)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), a "normalized" `BoW` text model,
  together with custom [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)
- [word embeddings for neural natural language processing](https://en.wikipedia.org/wiki/Word_embedding)

The implementation of the `BoW` classifiers is based on [Scikit-learn](https://scikit-learn.org/stable/getting_started.html).
The implementation of the `BERT` classifier is based on [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [PyTorch](https://pytorch.org/get-started/locally).

## File structure
### Folders (packages)
- *common*: General configuration and types.
- *models*: Saved `BoW` and `BERT` models.
- *pipeline*: Training a `BoW` text classifier based on a `Scikit-learn` pipeline.
- *tasks*: [Luigi](https://luigi.readthedocs.io/en/stable/) tasks and builders of classifiers leveraging [Optuna](https://optuna.readthedocs.io/en/stable/) and related utilities.
### The *message_spam_detection.ipynb* notebook
- Analysis of text classifiers already built by [Luigi](https://luigi.readthedocs.io/en/stable/) tasks.
### Other files
- Various configuration and setup files, such as *luigi.cfg*.
