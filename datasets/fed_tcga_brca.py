from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
    from flamby.datasets.fed_tcga_brca import metric, NUM_CLIENTS, Baseline, BaselineLoss
    from flamby.benchmarks.benchmark_utils import set_seed
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    # It's horrendous but frankly it works
    import sys
    import os
    sys.path.append(os.path.split(__file__)[0])
    from template_flamby_dataset import FLambyDataset
    


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(FLambyDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fed-TCGA-BRCA"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'test': ["test"],
        'seed': [42],
    }
    def __init__(self, *args, **kwargs):
        super().__init__(fed_dataset=FedDataset, model_arch=Baseline, loss=BaselineLoss, num_clients=NUM_CLIENTS, metric=metric, *args, **kwargs)