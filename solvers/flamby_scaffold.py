from benchopt import safe_import_context

from benchmark_utils.common import lrs, slrs
from benchmark_utils.template_flamby_strategy import FLambySolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from flamby.strategies import Scaffold


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(FLambySolver):
    """Implement the Scaffold FL strategy.

    By adding auxiliary variables or control variates (c) in the
    clients and server the authors of the related paper prove better
    bounds on the convergence assuming certain
    hypothesis. FLamby implements a faster version of Scaffold.
    The details can be found in FLamby.

    Parameters
    ----------
    FLambySolver : FlambySolver
        We define a common interface for all strategies implemented
        in FLamby.

    References
    ----------
    - https://arxiv.org/abs/1910.06378

    """

    # Name to select the solver in the CLI and to display the results.
    name = "Scaffold"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": lrs,
        "server_learning_rate": slrs,
        "batch_size": [32],
        "num_updates": [100],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(strategy=Scaffold, *args, **kwargs)

    def set_strategy_specific_args(self):
        self.strategy_specific_args = {"server_learning_rate": self.server_learning_rate}
