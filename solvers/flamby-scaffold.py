from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.strategies import Scaffold


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Scaffold'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'server_learning_rate': [0.0001, 0.001, 0.01],
        "batch_size": [32], # we deviate from flamby's formulation to be able to change batch-size in solver API
    }

    def set_objective(self, train_datasets, val_datasets, test_datasets, model):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark., 
        self.train_dataset, self.val_datasets, self.test_datasets, self.model = train_datasets, val_datasets, test_datasets, model

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        strat = Scaffold(self.train_datasets, self.model, learning_rate=self.learning_rate, server_learning_rate=self.server_learning_rate, nrounds=n_iter, batch_size=self.batch_size)
        m = strat.run()

        self.model = m

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model
