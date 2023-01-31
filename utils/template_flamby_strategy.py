from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from tqdm import tqdm

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
        "batch_size": [32],  # we deviate from flamby's fixed batch-size
        "num_updates": [100],
    }
    stopping_criterion = SufficientProgressCriterion(patience=100000000, strategy="callback")

    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def set_objective(self,
                      train_datasets,
                      val_datasets,
                      test_datasets,
                      model,
                      loss):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.,
        att_names = ["train_datasets",
                     "val_datasets",
                     "test_datasets",
                     "model",
                     "loss"]

        for att in att_names:
            setattr(self, att, eval(att))

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {}

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        self.train_dls = [
            dl(train_d, self.batch_size) for train_d in self.train_datasets
        ]
        self.set_strategy_specific_args()
        strat = self.strategy(
            self.train_dls,
            self.model,
            self.loss,
            SGD,
            self.learning_rate,
            self.num_updates,
            nrounds=-100, # It won't be used anyway as we do not call the run method
            **self.strategy_specific_args
        )
        # We take the first model, but we could return the full list for model
        # personalization
        # We are reproducing the run method but this time a callback checks stopping-criterion
        # at each round
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
