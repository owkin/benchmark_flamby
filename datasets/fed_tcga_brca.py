from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
    from flamby.datasets.fed_tcga_brca import (
        metric,
        NUM_CLIENTS,
        Baseline,
        BaselineLoss,
    )
    template_file_name = "template_flamby_dataset"
    FLambyDataset = import_ctx.import_from(template_file_name, "FLambyDataset")


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(FLambyDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fed-TCGA-BRCA"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {"train": ["pooled"], "test": ["test"], "seed": [42]}



    def __init__(self, *args, **kwargs):
        super().__init__(
            fed_dataset=FedDataset,
            model_arch=Baseline,
            loss=BaselineLoss,
            num_clients=NUM_CLIENTS,
            metric=metric,
            test_size=0.25,
            *args,
            **kwargs
        )
