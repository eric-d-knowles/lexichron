"""Word2vec training on the pivoted n-gram databases: the pieces shared by
the hyperparameter search (``lexichron tune``) and, later, the final
training with replicas.

- :mod:`grid`: hyperparameter axes, cells and their names.
- :mod:`corpus`: a year's corpus file written from the pivoted database.
- :mod:`trainer`: one model from one corpus file (gensim).
- :mod:`stability`: agreement between two models' nearest neighbours.
- :mod:`benchmarks`: similarity and analogy scores, vocabulary coverage.
"""
