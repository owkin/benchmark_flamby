import math

from benchopt.stopping_criterion import SufficientProgressCriterion


class CustomSPC(SufficientProgressCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        delta_objective_from_start = (
            start_objective - objective
        ) / start_objective  # noqa: E501
        if delta_objective_from_start < 0.0:
            self.debug(
                f"Exit with delta from start = {delta_objective_from_start:.2e}."
            )  # noqa: E501
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
