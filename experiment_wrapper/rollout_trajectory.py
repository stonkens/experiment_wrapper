from copy import copy
import tqdm
from experiment_wrapper.experiment import Experiment, ScenarioList
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


# FIXME: Requires testing!
# TODO: figure out how to have an argument that is either a torch.Tensor or a np.ndarray
# TODO: What makes more sense: Save all the data and then decide later what to plot? Or save only the data that is needed for plotting?
class RolloutTrajectory(Experiment):
    def __init__(
        self,
        name: str,
        start_x: np.ndarray,
        x_indices: Optional[List[int]] = None,
        x_labels: Optional[List[str]] = None,
        u_indices: Optional[List[int]] = None,
        u_labels: Optional[List[str]] = None,
        scenarios: Optional[ScenarioList] = None,
        n_sims_per_start: int = 5,
        t_sim: float = 5.0,
    ):
        """Initialize the rollout trajectory experiment. Optionally run multiple scenarios (e.g. different stochastic
        policies / disturbances / uncertain parameters) from multiple starting states

        Args:
            name (str): Name of the experiment
            start_x (np.ndarray): Starting states [n_start_states, n_dims]
            x_indices (List[int]): A list of the indices of state variables to log
            u_indices (List[int]): A list of the indices of control variables to log
            scenarios (Optional[ScenarioList], optional): ADifferent scenarios. Defaults to None.
            n_sims_per_start (int, optional): . Defaults to 5.
            t_sim (float, optional): _description_. Defaults to 5.0.
        """
        super().__init__(name)
        self.start_x = np.atleast_2d(start_x)
        self.x_indices = x_indices
        self.x_labels = x_labels
        self.u_indices = u_indices
        self.u_labels = u_labels
        self.scenarios = scenarios
        self.n_sims_per_start = n_sims_per_start  # For random disturbances
        self.t_sim = t_sim

    def set_idx_and_labels(self, dynamics):
        if self.x_indices is None:
            self.x_indices = list(range(dynamics.n_dims))
        # Default to saving all controls
        if self.u_indices is None:
            self.u_indices = list(range(dynamics.control_dims))
        if self.x_labels is None:
            self.x_labels = [dynamics.STATES[idi] for idi in self.x_indices]
        if self.u_labels is None:
            self.u_labels = [dynamics.CONTROLS[idi] for idi in self.u_indices]

    # TODO: Optional have multiple controllers too!
    def run(self, dynamics, controllers) -> pd.DataFrame:
        """At every time step:
        1) Check whether the control needs to be updated
        2) Save the current data
        3) Take step in the dynamics

        Args:
            controller ([type]): Class with callable methods (u) and optionally (is_unsafe, reset, V)

        Returns:
            pd.DataFrame: [description]
        """
        # Default to saving all state variables
        if not isinstance(controllers, list):
            controllers = [controllers]
        self.set_idx_and_labels(dynamics)

        results = []
        n_sims = self.n_sims_per_start * self.start_x.shape[0]
        x_sim_start = np.zeros((n_sims, dynamics.n_dims))

        for controller in controllers:
            for i in range(0, self.start_x.shape[0]):
                for j in range(0, self.n_sims_per_start):
                    x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

            x_current = x_sim_start
            if hasattr(controller, "reset_controller"):
                controller.reset(x_current)

            delta_t = dynamics.dt
            controller_update_freq = (
                int(controller.controller_dt / delta_t)
                if hasattr(controller, "controller_dt")
                else 1
            )
            num_steps = int(self.t_sim // delta_t)

            prog_bar_range = tqdm.tqdm(range(0, num_steps), desc="Controller rollout")

            for tstep in prog_bar_range:
                t = tstep * delta_t

                ######## UPDATE CONTROLLER ########
                if tstep % controller_update_freq == 0:
                    u_current = controller(x_current, t).reshape(n_sims, dynamics.control_dims)

                ########### LOGGING ###############
                for sim_index in range(n_sims):
                    base_log_packet = {"t": t}
                    base_log_packet["controller"] = (
                        controller.__class__.__name__
                        if hasattr(controller, "__class__")
                        else "Base"
                    )
                    base_log_packet["scenario"] = sim_index % self.n_sims_per_start
                    base_log_packet["rollout"] = sim_index // self.n_sims_per_start

                    if hasattr(
                        controller, "is_unsafe"
                    ):  # TODO: Have a call to controller that decides on what is loggable for each dict
                        base_log_packet["unsafe"] = controller.is_unsafe(
                            x_current[sim_index]
                        )  # TODO: Check where this call is made and make it optional

                    for i, state_index in enumerate(self.x_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.x_labels[
                            i
                        ]  # TODO: Maybe use dynamics attribute
                        log_packet["value"] = x_current[sim_index, state_index]
                        results.append(log_packet)

                    for i, control_index in enumerate(self.u_indices):
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = self.u_labels[i]
                        log_packet["value"] = u_current[sim_index, control_index]
                        results.append(log_packet)

                    if hasattr(
                        controller, "cbf"
                    ):  # TODO: have a call to controller that decides what measurements should also be logged
                        log_packet = copy(base_log_packet)
                        log_packet["measurement"] = "vf"
                        log_packet["value"] = controller.cbf.vf(x_current[sim_index])
                        results.append(log_packet)

                ########### SIMULATION ###############
                x_current = dynamics.step(x_current, u_current, t)

        return pd.DataFrame(results)


class TimeSeriesExperiment(RolloutTrajectory):
    def plot(
        self, dynamics, results_df: pd.DataFrame, display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        self.set_idx_and_labels(dynamics)
        sns.set_theme(context="talk", style="white", palette="colorblind")

        plot_V = "vf" in results_df.measurement.values
        num_plots = len(self.x_indices) + len(self.u_indices) + int(plot_V)

        fig, axs = plt.subplots(num_plots, 1, sharex=True)
        fig.set_size_inches(10, 4 * num_plots)
        for controller in results_df.controller.unique():
            for scenario in results_df.scenario.unique():
                for rollout in results_df.rollout.unique():
                    mask = (
                        (results_df.controller == controller)
                        & (results_df.scenario == scenario)
                        & (results_df.rollout == rollout)
                    )

                    for i, state_label in enumerate(self.x_labels):
                        ax = axs[i]
                        state_mask = mask & (results_df.measurement.values == state_label)
                        ax.plot(results_df[state_mask].t, results_df[state_mask].value)
                        ax.set_ylabel(state_label)

                    for i, control_label in enumerate(self.u_labels):
                        ax = axs[len(self.x_labels) + i]
                        control_mask = mask & (results_df.measurement.values == control_label)
                        ax.plot(results_df[control_mask].t, results_df[control_mask].value)
                        ax.set_ylabel(control_label)

                    if plot_V:
                        ax = axs[-1]
                        V_mask = mask & (results_df.measurement.values == "vf")
                        ax.plot(results_df[V_mask].t, results_df[V_mask].value)
                        ax.set_ylabel("$vf$")

        axs[-1].set_xlabel("t")

        fig_handle = ("Rollout (time series)", fig)

        if display_plots:
            plt.show()

        return [fig_handle]


class StateSpaceExperiment(RolloutTrajectory):
    def plot(
        self, dynamics, results_df: pd.DataFrame, display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        self.set_idx_and_labels(dynamics)
        assert len(self.x_labels) in [2, 3], "Can't plot in this dimension!"

        if len(self.x_labels) == 2:
            fig, ax = plt.subplots()
            fig.set_size_inches(9, 6)

            for controller in results_df.controller.unique():
                for scenario in results_df.scenario.unique():
                    for rollout in results_df.rollout.unique():
                        mask = (
                            (results_df.controller == controller)
                            & (results_df.scenario == scenario)
                            & (results_df.rollout == rollout)
                        )
                        ax.plot(
                            results_df[mask][self.x_labels[0]].value,
                            results_df[mask][self.x_labels[1]].value,
                        )
        else:
            raise NotImplementedError("Future work!")
