
Benchmarking FL strategies on FLamby's Fed-TCGA
===============================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solve FLamby's Fed-TCGA dataset task, which is 
survival in Federated Settings. The goal is to maximize the mean accuracy on the
test clients:


$$\\max_{\theta} \\sum_{k=0}^{5} Acc(f_{\theta}(X_{k}), y_{k})$$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times 39} \\ , \\quad \theta \\in \\mathbb{R}^40$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/owkin/benchopt_flamby_tcga
   $ benchopt run benchopt_flamby_tcga

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchopt_flamby_tcga -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/owkin/benchopt_flamby_tcga/workflows/Tests/badge.svg
   :target: https://github.com/owkin/benchopt_flamby_tcga/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
