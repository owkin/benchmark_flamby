import math
import types

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from torch.utils.data import DataLoader as dl
    from torch.optim import SGD


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class FLambySolver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = "Strategy"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [0.01],
        "batch_size": [32],  # we deviate from flamby's dataset specific batch-size  # noqa: E501
        "num_updates": [100],
    }
    stopping_criterion = SufficientProgressCriterion(patience=100000000, strategy="callback")   # noqa: E501

    # Ok so this is a pity because I would like my stopping criterion to
    # have a custom check_convergence method and because of that I need to
    # go down the rabbit hole. The way to do that cleanly would be to use
    # class inheritance and to define another stopping_criterion class.
    # However I cannot fix the pickling issues associated with using an
    # inherited class no matter what I do ...
    # Therefore I need to use one of the original Benchopt stopping_criterion
    # classes. Thus I have to modify dynamically the original class's
    # instance's check_convergence method. However to make it harder, this
    # instance is not the one that will be called inside the run method
    # instead another instance of the original class is used: the one
    # created by the get_runner_instance method so I need to modify the
    # get_runner_instance method of the original class's instance so
    # that it produces a modified instance of class with the proper
    # check_convergence method. This explains the following dark magic.

    stopping_criterion.get_runner_instance_original = stopping_criterion.get_runner_instance   # noqa: E501

    def decorated_get_runner_instance(self, *args, **kwargs):
        res = self.get_runner_instance_original(*args, **kwargs)

        def check_convergence(self, objective_list):
            """Check if the solver should be stopped based on the objective
            curve.
            Parameters
            ----------
            objective_list : list of dict
                List of dict containing the values associated to the objective
                at each evaluated points.
            Returns
            -------
            stop : bool
                Whether or not we should stop the algorithm.
            progress : float
                Measure of how far the solver is from convergence.
                This should be in [0, 1], 0 meaning no progress and 1 meaning
                that the solver has converged.
            """
            # Compute the current objective and update best value
            start_objective = objective_list[0][self.key_to_monitor]

            objective = objective_list[-1][self.key_to_monitor]

            # We exit if one value of the objective is lower than the
            # starting point. This is a bit random but it serves as a
            # divergence check of some sort
            delta_objective_from_start = (start_objective - objective) / start_objective    # noqa: E501
            if delta_objective_from_start < 0.:
                self.debug(f"Exit with delta from start = {delta_objective_from_start:.2e}.")   # noqa: E501
                return True, 1

            delta_objective = self._best_objective - objective
            delta_objective /= abs(objective_list[0][self.key_to_monitor])

            # Store only the last ``patience`` values for progress
            self._progress.append(delta_objective)
            if len(self._progress) > self.patience:
                self._progress.pop(0)

            delta = max(self._progress)
            if delta <= self.eps * self._best_objective:
                self.debug(f"Exit with delta = {delta:.2e}.")
                return True, 1

            progress = math.log(max(abs(delta), self.eps)) / math.log(self.eps)
            return False, progress

        res.check_convergence = types.MethodType(check_convergence, res)
        return res

    stopping_criterion.get_runner_instance = types.MethodType(decorated_get_runner_instance, stopping_criterion)    # noqa: E501

    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        # We override dynamically the method of the instance, inheritance
        # would be cleaner but
        # I could not make it work because of pickling issues
        # We basically do not stop unless something goes terribly wrong

    def set_objective(self,
                      train_datasets,
                      test_datasets,
                      collate_fn,
                      is_validation,
                      model,
                      loss):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.,
        att_names = ["train_datasets",
                     "test_datasets",
                     "collate_fn",
                     "is_validation",
                     "model",
                     "loss"]

        for att in att_names:
            setattr(self, att, eval(att))

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {}

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of rounds
        # (max_runs * 10)

        self.train_dls = [
            dl(train_d, self.batch_size, collate_fn=self.collate_fn) for train_d in self.train_datasets   # noqa: E501
        ]
        self.set_strategy_specific_args()
        strat = self.strategy(
            self.train_dls,
            self.model,
            self.loss,
            SGD,
            self.learning_rate,
            self.num_updates,
            nrounds=-100,  # It won't be used anyway as we do not call the run method   # noqa: E501
            **self.strategy_specific_args
        )
        # We are reproducing the run method but this time a callback checks
        # stopping-criterion at each round, which allows to cache computations
        # and do a single run
        while callback(strat.models_list[0].model):
            strat.perform_round()

        self.final_model = strat.models_list[0].model

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.final_model

    # Not used if callback is used
    @staticmethod
    def get_next(stop_val):
        """This function gives the sampling rate of the curve.
        """
        return stop_val + 10
