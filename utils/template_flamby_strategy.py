from benchopt import BaseSolver, safe_import_context

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
    name = 'Strategy'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'learning_rate': [0.01],
        "batch_size": [32], # we deviate from flamby's formulation to be able to change batch-size in solver API
        "num_updates": [100],
    }
    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def set_objective(self, train_datasets, val_datasets, test_datasets, model, loss):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark., 
        self.train_datasets, self.val_datasets, self.test_datasets, self.model, self.loss = train_datasets, val_datasets, test_datasets, model, loss

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {}

    def run(self, n_iter=50):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        self.train_dls = [dl(train_d, batch_size=self.batch_size) for train_d in self.train_datasets]
        self.set_strategy_specific_args()
        strat = self.strategy(self.train_dls, self.model, self.loss, SGD, self.learning_rate, self.num_updates, nrounds=n_iter, **self.strategy_specific_args)
        # We take the first model, but we could return the full list for model personalization
        m = strat.run()[0]

        self.model = m

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model