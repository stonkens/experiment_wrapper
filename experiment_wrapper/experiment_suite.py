"""A collection suite manages a collection of experiments, allowing the user to run all of them in a single run."""

from cbf_opt.experiment.experiment import Experiment
from typing import List, Optional, Tuple
import pandas as pd
import os, datetime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


class ExperimentSuite:
    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments

    def run_all(self, controller_under_test: "Controller") -> List[pd.DataFrame]:
        results = []
        for experiment in self.experiments:
            results.append(experiment.run(controller_under_test))
        return results

    def run_all_and_save_to_csv(self, controller_under_test: "Controller", save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        subdir = f"{save_dir}/{timestamp}"  # Save all experiments to subdirectory
        for experiment in self.experiments:
            experiment.run_and_save_to_csv(controller_under_test, subdir)

    def run_all_and_plot(
        self, controller_under_test: "Controller", display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        fig_handles = []
        for experiment in self.experiments:
            fig_handles += experiment.run_and_plot(controller_under_test, display_plots)
        return fig_handles
