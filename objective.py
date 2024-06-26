import re
from itertools import zip_longest

from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    import gc
    from flamby.benchmarks.benchmark_utils import set_seed
    from flamby.datasets.fed_kits19 import FedKits19, evaluate_dice_on_tests
    from flamby.datasets.fed_lidc_idri import (
        FedLidcIdri,
        evaluate_dice_on_tests_by_chunks,
    )
    from flamby.utils import evaluate_model_on_tests
    from torch.utils.data import DataLoader as dl


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    """Compute per-client and global metrics and losses.

    This objective computes several metrics of interests based on FLamby losses
    and Flamby metrics per dataset and globally.
    This way we can compare per-dataset metric vs averaged metrics across
    datasets and global metric when datasets are pooled.
    The objective_value is the average train or test loss across datasets
    this depends on wether we use the validation datasets or the test
    ones.


    Parameters
    ----------
    BaseObjective : BaseObjective
        The base objective in BenchOpt

    """

    # Name to select the objective in the CLI and to display the results.
    name = "FLamby"

    # Create a link in the result files from https://benchopt.github.io/results
    url = "https://github.com/owkin/benchmark_flamby"

    # Make it easy to install the benchmark
    install_cmd = "conda"
    requirements = [
        "pip:git+https://github.com/owkin/FLamby#egg=flamby[all_extra]"
    ]

    parameters = {
        "seed": [42],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.4"

    def set_data(
        self,
        train_datasets,
        test_datasets,
        is_validation,
        pooled_test_dataset,
        model_arch,
        metric,
        loss,
        num_clients,
        batch_size_test,
        collate_fn,
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        att_names = [
            "train_datasets",
            "test_datasets",
            "is_validation",
            "pooled_test_dataset",
            "model_arch",
            "metric",
            "loss",
            "num_clients",
            "batch_size_test",
            "collate_fn",
        ]
        for att in att_names:
            setattr(self, att, eval(att))

        # We init the model
        set_seed(self.seed)
        self.model = self.model_arch()
        # Small boilerplate for datasets that require custom evaluation
        if hasattr(train_datasets[0], "dataset"):
            check_dataset = train_datasets[0].dataset
        else:
            check_dataset = train_datasets[0]
        if isinstance(check_dataset, FedLidcIdri):

            def evaluate_func(m, test_dls, metric):
                return evaluate_dice_on_tests_by_chunks(m, test_dls)

            self.eval = evaluate_func

        elif isinstance(check_dataset, FedKits19):

            def evaluate_func(m, test_dls, metric):
                return evaluate_dice_on_tests(m, test_dls, metric)

            self.eval = evaluate_func
        else:
            self.eval = evaluate_model_on_tests

    def compute_avg_loss_on_client(self, model, dataset):
        average_loss = 0.0
        count_batch = 0
        for X, y in dl(dataset, self.batch_size_test, shuffle=False, collate_fn=self.collate_fn):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            average_loss += self.loss(model(X), y).item()
            count_batch += 1
        del X, y
        average_loss /= float(count_batch)
        return average_loss

    def evaluate_result(self, model):
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        test_dls = [
            dl(
                test_d,
                self.batch_size_test,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for test_d in self.test_datasets  # noqa: E501
        ]

        def robust_metric(y_true, y_pred):
            try:
                return self.metric(y_true, y_pred)
            except (ValueError, ZeroDivisionError):
                return np.nan

        if self.is_validation:
            test_name = "val"
        else:
            test_name = "test"

        # Evaluation on the different test sets
        res = self.eval(model, test_dls, robust_metric)

        # Evaluation on the pooled test set
        pooled_res_value = self.eval(
            model,
            [
                dl(
                    self.pooled_test_dataset,
                    self.batch_size_test,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                )
            ],
            robust_metric,
        )[
            "client_test_0"
        ]  # noqa: E501

        # We do not take into account clients where metric is not defined
        # nd use the average metric across clients as the default benchopt
        # metric "value". Note that we weigh all clients equally
        average_metric = 0.0
        nb_clients_nan = 0
        skip_clients = []
        for k, _ in res.copy().items():
            single_client_metric = res.pop(k)
            if np.isnan(float(single_client_metric)):
                skip_clients.append(int(re.findall("[0-9]+", k)[0]))
                nb_clients_nan += 1
                continue
            average_metric += single_client_metric
            res[k + "_" + test_name + "_metric"] = single_client_metric

        average_metric /= float(len(res.keys()) - nb_clients_nan)

        res["average_" + test_name + "_metric"] = average_metric

        res["pooled_" + test_name + "_metric"] = pooled_res_value

        # We also compute average losses on batches on the different clients
        # both on test and train
        average_test_loss = 0.0
        average_train_loss = 0.0
        for idx, (train_d, test_d) in enumerate(
            zip_longest(self.train_datasets, self.test_datasets)
        ):
            if train_d is not None:
                cl_train_loss = self.compute_avg_loss_on_client(model, train_d)
                res[f"train_loss_client_{idx}"] = cl_train_loss
                average_train_loss += cl_train_loss

            # If metrics is not defined then the test loss should not be
            # computed
            if idx in skip_clients:
                continue

            if test_d is not None:
                cl_test_loss = self.compute_avg_loss_on_client(model, test_d)
                res[test_name + f"_loss_client_{idx}"] = cl_test_loss
                average_test_loss += cl_test_loss

        # We compute average loss on test if it doesn't exist already
        if len(self.test_datasets) > 1:
            pooled_test_loss = self.compute_avg_loss_on_client(
                model, self.pooled_test_dataset
            )  # noqa: E501
        else:
            pooled_test_loss = res[test_name + "_loss_client_0"]

        res["average_" + test_name + "_loss"] = pooled_test_loss

        num_train_sets = len(self.train_datasets)
        num_test_sets = len(self.test_datasets)
        # We compute average losses across clients, weighting clients equally
        average_train_loss /= float(num_train_sets)
        average_test_loss /= float(num_test_sets - nb_clients_nan)
        res["average_train_loss"] = average_train_loss
        res["average_" + test_name + "_loss"] = average_test_loss

        # Important for display purposes, this way averages are displayed first
        sorted_res = {key: value for key, value in sorted(res.items())}

        # Very important because of check_convergence that operates on val
        # if is validation or on train
        if self.is_validation:
            objective_value = average_test_loss
        else:
            objective_value = average_train_loss

        keys_list = list(sorted_res.keys())

        # We add the value that is used in convergence_check in the first
        # place so that it appears by default and the rest of the values are
        # sorted in the clickable list
        new_res = {}
        new_res["value"] = objective_value
        for k in keys_list:
            new_res[k] = sorted_res.pop(k)

        gc.collect()
        torch.cuda.empty_cache()
        return new_res

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return dict(model=self.model_arch())

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            train_datasets=self.train_datasets,
            test_datasets=self.test_datasets,
            collate_fn=self.collate_fn,
            is_validation=self.is_validation,
            model=self.model,
            loss=self.loss,
        )
