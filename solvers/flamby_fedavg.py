from benchopt import safe_import_context

from benchmark_utils.common import lrs
from benchmark_utils.template_flamby_strategy import FLambySolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.strategies import FedAvg


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(FLambySolver):
    """Implement the FedAvg strategy.

    This solver uses FLamby's implementation of FedAvg.

    Parameters
    ----------
    FLambySolver : FlambySolver
        We define a common interface for all strategies implemented
        in FLamby.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    """

    # Name to select the solver in the CLI and to display the results.
    name = "FederatedAveraging"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": lrs,
        "batch_size": [32],
        "num_updates": [100],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(strategy=FedAvg, *args, **kwargs)
