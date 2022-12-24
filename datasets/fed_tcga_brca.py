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
    


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fed-TCGA-BRCA"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'test': ["val", "test"],
        'seed': [42],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # dataset independent part
        set_seed(self.seed)
        self.num_clients = NUM_CLIENTS
        self.trainval_datasets = [FedDataset(center=i, train=True) for i in range(self.num_clients)]
        self.trainval_sizes = [len(d) for d in self.trainval_datasets]
        self.test_datasets = [FedDataset(center=i, train=False) for i in range(self.num_clients)]

        if self.test == "val":
            # This part may vary across datasets specifically for label/RAM issues here we separate for each client a validation
            # set while stratifying wrt the target variable, in this case censorship
            self.trainval_indices_list = [train_test_split(range(size), stratify=[float(e[i][1][0]) for i in range(size)], random_state=self.seed) for e, size in zip(self.trainval_datasets, self.trainval_sizes)]
            self.train_datasets = [Subset(e, trainval_indices[0]) for e, trainval_indices in zip(self.trainval_datasets, self.trainval_indices_list)]
            self.val_datasets = [Subset(e, trainval_indices[1]) for e, trainval_indices in zip(self.trainval_datasets, self.trainval_indices_list)]
            self.test_datasets = self.val_datasets

        elif self.test == "test":
            pass
        else:
            raise ValueError()

        # ! The metric depends on the dataset it has to be passed to the objective,
        # same for loss and model

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(train_datasets=self.train_datasets, val_datasets=self.val_datasets, test_datasets=self.test_datasets, model_arch=Baseline, metric=metric, loss=BaselineLoss)
