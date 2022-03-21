"""Defines a gneeric experiment that can be extended to any type of controls problem.

"Experiment" is anything that tests the behavior of a controller, and should only be limited to a single function, e.g., simulating a rollout or plotting the Lyapunov function on a grid



Each experiment should do the following:
    1. Run the experiment on a given controller
    2. Save the results of that experiment to a CSV (tidy data principle) https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html
    3. Plot the results of the experiment and return the plot handle, with an option of displaying the plot
    """

from abc import ABCMeta, abstractmethod
import numpy as np

from typing import List, Tuple, Dict
import pandas as pd
from matplotlib.pyplot import figure

Scenario = Dict[str, float]
ScenarioList = List[Scenario]


class Experiment(metaclass=ABCMeta):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self, dynamics, controllers):
        """
        Run the experiment on a given controller

        returns:
            a pandas Dataframe containing the result of the experiment (each row corresponding to a single observation from the experiment)
        """
        raise NotImplementedError("run() must be implemented by a subclass")

    def plot(self, dynamics, results_df, display_plots: bool = False) -> List[Tuple[str, figure]]:
        """[summary]

        Args:
            dynamics (Dynamics): Dynamics object
            display_plots (bool, optional): If True, display the plots (block until user responds). Defaults to False.

        Returns:
            List[Tuple[str, figure]]: Contains name of each figures and the figure object
        """
        raise NotImplementedError("plot() must be implemented by a subclass")

    def run_and_save_to_csv(self, dynamics, controllers, save_dir: str):
        import os

        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{self.name}.csv"

        results = self.run(dynamics, controllers)
        results.to_csv(filename, index=False)

    def run_and_plot(self, dynamics, controllers, display_plots: bool = False):
        results_df = self.run(dynamics, controllers)
        return self.plot(dynamics, results_df, display_plots)