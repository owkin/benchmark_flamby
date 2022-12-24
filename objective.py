from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    #from flamby.strategies import FedAveraging, FedProx, Scaffold, FedAdam
    from flamby.benchmarks.benchmark_utils import set_seed
    from flamby.utils import evaluate_model_on_tests
    from torch.utils.data import DataLoader


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "FLamby Average Metric across client"
    def __init__(self):
        super().__init__()
        self.seed = 42

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, train_datasets, val_datasets, test_datasets, model_arch, metric, loss):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.train_dataset, self.val_datasets, self.test_datasets, self.model_arch, self.metric, self.loss = train_datasets, val_datasets, test_datasets, model_arch, metric, loss
        # We init the model
        set_seed(self.seed)
        self.model = self.model_arch()


    def compute(self, model):
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        m = evaluate_models_on_tests(self.test_datasets, model, self.metric)

        # We assume everything is separable
        average_metric = 0.
        for _, v in m.items():
            average_metric += v
        average_metric /= float(self.num_clients)

        average_test_loss = 0.
        average_train_loss = 0.
        for train_d, test_d in zip(self.train_datasets, self.test_datasets):
            single_client_train_loss = 0.
            count_batch = 0
            for X, y in DataLoader(train_d, batch_size=self.batch_size, shuffle=False):
                single_client_train_loss += self.loss(model(X), y)
                count_batch += 1
            single_client_train_loss /= float(count_batch)
            average_train_loss += single_client_train_loss

            single_client_test_loss = 0.
            count_batch = 0
            for X, y in DataLoader(test_d, batch_size=self.batch_size, shuffle=False):
                single_client_test_loss += self.loss(model(X), y)
                count_batch += 1
            single_client_test_loss /= float(count_batch)
            average_test_loss += single_client_test_loss
        average_test_loss /= float(self.num_clients)
        average_train_loss /= float(self.num_clients)

        return dict(value=average_metric, average_train_loss=average_train_loss, average_test_loss=average_test_loss)

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return self.model_arch()

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            train_datasets=self.train_datasets,
            val_datasets=self.val_datasets,
            test_datasets=self.test_datasets,
            model=self.model,
        )
