from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.benchmarks.benchmark_utils import set_seed
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class FLambyDataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fed-Dataset"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {"test": ["test", "val"]}

    def __init__(
        self,
        fed_dataset,
        model_arch,
        loss,
        num_clients,
        metric,
        seed=42,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # We choose to define the test batch-size in the dataset as it is
        # heavily dataset dependent
        self.batch_size_test = 100
        self.fed_dataset = fed_dataset
        self.model_arch = model_arch
        self.loss = loss
        self.num_clients = num_clients
        self.metric = metric
        # For train/validation-splits and possibly dataset creation if needed
        self.seed = seed

    def train_test_split_datasets(self):
        # This part may vary across datasets specifically for label/RAM issues
        # here we separate for each client a validation
        # set while stratifying wrt the target variable, in this case
        # censorship. Therefore one could have to reimplement it
        self.trainval_indices_list = [
            train_test_split(
                range(size),
                stratify=[float(e[i][1][0]) for i in range(size)],
                random_state=self.seed,
            )
            for e, size in zip(self.trainval_datasets, self.trainval_sizes)
        ]
        self.train_datasets = [
            Subset(e, trainval_indices[0])
            for e, trainval_indices in zip(
                self.trainval_datasets, self.trainval_indices_list
            )
        ]
        self.val_datasets = [
            Subset(e, trainval_indices[1])
            for e, trainval_indices in zip(
                self.trainval_datasets, self.trainval_indices_list
            )
        ]
        self.test_datasets = self.val_datasets

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # dataset independent part that shouldn't have to be reimplemented
        set_seed(self.seed)
        self.trainval_datasets = [
            self.fed_dataset(i, train=True) for i in range(self.num_clients)
        ]
        self.trainval_sizes = [len(d) for d in self.trainval_datasets]
        self.test_datasets = [
            self.fed_dataset(i, train=False) for i in range(self.num_clients)
        ]

        if self.test == "val":
            self.train_test_split_datasets()

        elif self.test == "test":
            self.train_datasets = self.trainval_datasets
            self.val_datasets = None
        else:
            raise ValueError()

        # ! The metric depends on the dataset it has to be passed to the
        # objective, same for loss and model

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_datasets=self.train_datasets,
            val_datasets=self.val_datasets,
            test_datasets=self.test_datasets,
            model_arch=self.model_arch,
            metric=self.metric,
            loss=self.loss(),
            num_clients=self.num_clients,
            batch_size_test=self.batch_size_test,
        )
