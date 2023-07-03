
Benchmarking FL strategies on FLamby with benchopt
==================================================
|Build Status| |Python 3.6+|


.. image:: https://github.com/owkin/FLamby/blob/main/docs/logo.png
   :scale: 50%
   :width: 20px
   :target: https://owkin.github.io/FLamby/

This benchmark is dedicated to tuning cross-silo FL strategies on Flamby_'s datasets.
The goal is to maximize the average metric across clients using each provided model
on the val/test clients:


$$\\max_{\\theta} \\sum_{k=0}^{K} m(f_{\\theta}(X_{k}), y_{k})$$


where $K$ stands for the number of clients participating in the
Federated Learning training, $p$ (or ``n_features``) stands for the number of features
, :math:`$\theta$` the parameters of the model of dimension $N$,
$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad \\theta \\in \\mathbb{R}^N$$
and $m$, the metric of interest.
To ease comparison, we fix the number of local updates to 100 and the maximum number of rounds
to 120 (12*10).

**Try to beat the FLamby by adding your own solver !**  

You can even use your favorite python FL-frameworks such as substra_ or FedBioMed_ to build your solver !


Install
-------

First go to Flamby_ and install it using the following commands (see the API Doc_ if needed): 

.. code-block::

   $ git clone https://github.com/owkin/FLamby.git
   $ cd FLamby
   $ conda create -n benchmark_flamby
   $ conda activate benchmark_flamby
   $ pip install -e ".[all_extra]" # Note that the all_extra option installs all dependencies for all 7 datasets

This benchmark can then be run on Fed-TCGA-BRCA's validation sets using the following commands, which will launch
a grid-search on all parameters found in `utils/common.py` for the FederatedAveraging strategy doing 120 rounds
(`--max-runs 12` * 10) with 100 local updates per round:  

.. code-block::

   $ pip install -U benchopt
   $ cd ..
   $ git clone https://github.com/owkin/benchmark_flamby
   $ cd benchmark_flamby
   $ benchopt run --timeout 24h --max-runs 12 -s FederatedAveraging -d Fed-TCGA-BRCA

To test a specific value of hyper-parameters just fill a yaml config file with the appropriate hyper-parameters for each solver
following the `example_config.yml` example config file.  

.. code-block::

   $ benchopt run --config ./example_config.yml

Or use directly the CLI:

.. code-block::

   $ benchopt run -s FederatedAveraging[batch_size=32,learning_rate=0.031622776601683794]


For the whole benchmark on Fed-TCGA-BRCA we successively run all hyper-parameters of the grid for all strategies.
To reproduce results just launch the following command (note that it takes several hours to complete but can be cached):  

.. code-block::

   $ bash launch_validation_benchmarks.sh

This script should reproduce the html plot visible on the results for Fed-TCGA-BRCA and produce a config with all best validation hyper-parameters
for each strategy.

To produce the final plot on the test run:  

.. code-block::

   $ benchopt run --timeout 24h --config ./best_config_test_Fed-TCGA-BRCA.yml

To benchmark on other datasets of FLamby, follow FLamby's instructions to download each dataset, for example you can
find Fed-Heart-Disease's download's instructions here_.
Then once the dataset is downloaded one can run the same commands changing the dataset argument i.e.:  

For the validation:

.. code-block::

   $ bash launch_validation_benchmarks.sh Fed-Heart-Disease

For the results on the test sets:  

.. code-block::

   $ benchopt run --timeout 24h --config ./best_config_found_for_heart_disease.yml


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/owkin/benchmark_flamby/workflows/Tests/badge.svg
   :target: https://github.com/owkin/benchmark_flamby/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/

    
.. _Flamby: https://github.com/owkin/FLamby
    
.. _Doc: https://owkin.github.io/FLamby/

.. _here: https://owkin.github.io/FLamby/fed_heart.html#download-and-preprocessing-instructions

.. _substra: https://github.com/Substra/substrafl

.. _FedBioMed: https://gitlab.inria.fr/fedbiomed/fedbiomed


FAQ
---
.. code-block::
   $ ModuleNotFoundError: No module named 'flamby.whatever' 

Make sure that benchopt CLI uses the right Python interpreter. 
To do that one might have to do `conda init bash` to put conda path in the PATH.
