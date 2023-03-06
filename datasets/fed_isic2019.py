from benchopt import safe_import_context
from benchmark_utils.template_flamby_dataset import FLambyDataset


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.datasets.fed_isic2019 import FedIsic2019 as FedDataset
    from flamby.datasets.fed_isic2019 import (
        metric,
        NUM_CLIENTS,
        Baseline,
        BaselineLoss,
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(FLambyDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fed-ISIC2019"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {"train": ["fl"], "test": ["val"], "seed": [42]}

    def __init__(self, *args, **kwargs):

        def stratify_on_y(sample):
            return sample[1]

        super().__init__(
            fed_dataset=FedDataset,
            model_arch=Baseline,
            loss=BaselineLoss,
            num_clients=NUM_CLIENTS,
            metric=metric,
            test_size=0.25,
            stratify_func=stratify_on_y,
            *args,
            **kwargs
        )
