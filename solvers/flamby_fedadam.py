from benchopt import safe_import_context

from benchmark_utils.common import lrs, slrs
from benchmark_utils.template_flamby_strategy import FLambySolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.strategies import FedAdam


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(FLambySolver):
    """Implement the FedAdam FL strategy.

    This solver uses FLamby's implementation of the federation
    of the Adam solver as described in Reddi et al. 2020.

    Parameters
    ----------
    FLambySolver : FlambySolver
        We define a common interface for all strategies implemented
        in FLamby.

    References
    ----------
    - https://arxiv.org/abs/1412.6980
    - https://arxiv.org/abs/2003.00295

    """

    # Name to select the solver in the CLI and to display the results.
    name = "FedAdam"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": lrs,
        "server_learning_rate": slrs,
        "batch_size": [32],
        "tau": [1e-8],
        "beta1": [0.9],
        "beta2": [0.999],
        "num_updates": [100],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(strategy=FedAdam, *args, **kwargs)

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {
            "server_learning_rate": self.server_learning_rate,
            "tau": self.tau,
            "beta1": self.beta1,
            "beta2": self.beta2,
        }
