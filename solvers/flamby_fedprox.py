from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.strategies import FedProx
    from torch.utils.data import DataLoader as dl
    from torch.optim import SGD
    FLambySolver = import_ctx.import_from(
        'template_flamby_strategy', 'FLambySolver'
    )



# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(FLambySolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'FedProx'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'learning_rate': [0.01],
        "batch_size": [32], # we deviate from flamby's formulation to be able to change batch-size in solver API
        "num_updates": [100],
        "nrounds": [10],
        "mu": [0.001],
    }
    def __init__(self, *args, **kwargs):
        super().__init__(strategy=FedProx, *args, **kwargs)

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {"mu": self.mu}
