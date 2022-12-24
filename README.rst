
Benchmarking FL strategies on FLamby with benchopt
==================================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to tuning FL strategies on FLamby's datasets.
The goal is to maximize the average metric using each provided model
on the val/test clients:


$$\\max_{\\theta} \\sum_{k=0}^{K} m(f_{\\theta}(X_{k}), y_{k})$$


where $K$ stands for the number of clients participating in the
Federated Learning training, $p$ (or ``n_features``) stands for the number of features
, $ \\theta $ the parameters of the model of dimension $N$,
$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad \\theta \\in \\mathbb{R}^N$$
and $m$, the metric of interest.


Install
--------

This benchmark can be run using the following commands:

.. code-block::
   $ git clone https://github.com/owkin/FLamby.git
   $ cd FLamby
   $ conda create -n benchmark_flamby
   $ conda activate benchmark_flamby
   $ pip install -e ".[all_extra]"
   $ pip install -U benchopt
   $ cd ..
   $ git clone https://github.com/owkin/benchmark_flamby
   $ benchopt run benchmark_flamby

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_flamby -s scaffold -d fed_tcga_brca --max-runs 10 --n-repetitions 2


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/owkin/benchmark_flamby/workflows/Tests/badge.svg
   :target: https://github.com/owkin/benchmark_flamby/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/

FAQ
---
.. code-block::
   $ ModuleNotFoundError: No module named 'flamby.whatever' 

Make sure that benchopt CLI uses the right Python interpreter. 
To do that one might have to do `conda init bash` to put conda path in the PATH.