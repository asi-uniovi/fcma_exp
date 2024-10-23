"""Module to carry out comparisons between Conlloovia and FCMA solutions."""

from collections import defaultdict
import logging
import math
from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from rich.table import Table
from rich.logging import RichHandler

from pulp import PULP_CBC_CMD  # type: ignore

import fcma

from conlloovia import (
    Allocation,
    App,
    ConllooviaAllocator,
    ContainerClass,
    InstanceClass,
    LimitsAdapter,
    Problem,
    Solution,
    Status,
    System,
    Workload,
)
from conlloovia.first_fit import FirstFitAllocator, FirstFitIcOrdering

from cloudmodel.unified.units import CurrencyPerTime, Requests, Time

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

FcmaSolutionDict = Dict[Tuple[int, float], fcma.Solution]  # [speed, sfmpl] -> Solution

# Several FCMA speeds and several SFMPLS for each can be defined here in a list
SPEEDS = [1]
SFMPLS = [0.5]


class ComparisonResult:
    """Comparison result between Conlloovia and FCMA solutions. It contains the solutions.
    '_w2' means 'second window', i.e., the solution for a second window that is only
    used for the recycling metric."""

    def __init__(
        self,
        conlloovia_sol: Solution,
        conlloovia_sol_w2: Optional[Solution],
        fcma_sols: FcmaSolutionDict,
        fcma_sols_w2: FcmaSolutionDict,
        ffc_sol: Solution,
        ffc_sol_w2: Optional[Solution],
        ffp_sol: Solution,
        ffp_sol_w2: Optional[Solution],
        par_names: list[str],
        par_values: list[float],
    ):
        self.par_names = par_names
        self.par_values = par_values
        self.conlloovia_sol = conlloovia_sol
        self.conlloovia_sol_w2 = conlloovia_sol_w2
        self.fcma_sols = fcma_sols
        self.fcma_sols_w2 = fcma_sols_w2
        self.ffc_sol = ffc_sol
        self.ffc_sol_w2 = ffc_sol_w2
        self.ffp_sol = ffp_sol
        self.ffp_sol_w2 = ffp_sol_w2

        # Fields to store the metrics before removing the allocations. There will
        # be an entry for Conlloovia, FFC and FFP for each SMPFL
        self.fault_tolerance_m = {}
        self.container_isolation_m = {}
        self.vm_recycling_m = {}
        self.vm_load_balance_m = {}

    def sol_cost(self, sol: Solution) -> CurrencyPerTime:
        """Return the cost per time in usd/hour for a Conlloovia-like solution."""
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return np.nan * CurrencyPerTime("usd/hour")
        sched_time = sol.problem.sched_time_size
        cost_per_time = sol.cost / sched_time
        return cost_per_time.to("usd/hour")

    def conlloovia_cost(self) -> CurrencyPerTime:
        """Return the cost per time for the Conlloovia solution."""
        return self.sol_cost(self.conlloovia_sol)

    def costs(
        self,
    ) -> tuple[CurrencyPerTime, ...]:
        """Return the costs per time for the Conlloovia, FFC, FFP and FCMA solutions."""
        return (
            self.sol_cost(self.conlloovia_sol),
            self.sol_cost(self.ffc_sol),
            self.sol_cost(self.ffp_sol),
            *(self.fcma_cost(speed, sfmpl) for speed in SPEEDS for sfmpl in SFMPLS),
        )

    def sol_cost_str(self, sol: Solution) -> str:
        """Return the cost per time for a Conlloovia-like solution as a string."""
        cost = self.sol_cost(sol)
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return "N/A"
        return f"{cost.magnitude:10.4f}"

    def costs_str(self) -> tuple[str, ...]:
        """Return the costs per time for the Conlloovia, FFC, FFP and FCMA solutions as
        strings."""
        costs = self.costs()
        conlloovia_cost_str = self.sol_cost_str(self.conlloovia_sol)
        ffc_cost_str = self.sol_cost_str(self.ffc_sol)
        ffp_cost_str = self.sol_cost_str(self.ffp_sol)

        return (
            conlloovia_cost_str,
            ffc_cost_str,
            ffp_cost_str,
            *[f"{cost.magnitude:10.4f}" for cost in costs[3:]],
        )

    def fcma_cost(self, speed: int, sfmpl: float) -> CurrencyPerTime:
        """Return the cost per time for the FCMA solution at the given speed and sfmpl."""
        return self.fcma_sols[speed, sfmpl].statistics.final_cost

    def diff_cost_fcma_conlloovia_str(self, speed: int, sfmpl: float) -> str:
        """Return the difference between the FCMA and Conlloovia costs as a string. If
        Conlloovia didn't find a solution, return 'N/A'."""
        if self.conlloovia_status() not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return "N/A"

        fcma_cost = self.fcma_cost(speed, sfmpl)
        return f"{(fcma_cost - self.conlloovia_cost()).magnitude:10.4f}"

    def total_secs(self) -> tuple[float, ...]:
        """Return the solving times in seconds for the Conlloovia and FCMA solutions."""
        c_stats = self.conlloovia_sol.solving_stats
        conlloovia_time = c_stats.creation_time + c_stats.solving_time
        return (
            conlloovia_time,
            *(
                self.fcma_sols[speed, sfmpl].statistics.total_seconds
                for speed in SPEEDS
                for sfmpl in SFMPLS
            ),
        )

    def conlloovia_vars(self) -> int:
        """Return the number of variables in the Conlloovia solution."""
        return self.conlloovia_sol.num_vars_x() + self.conlloovia_sol.num_vars_z()

    def conlloovia_status(self) -> Status:
        """Return the status of the Conlloovia solution."""
        return self.conlloovia_sol.solving_stats.status

    def c_lower_bound_d_h(self) -> float:
        """Return the lower bound of the solution obtained by Conlloovia in dollars per
        hour."""
        window_lower_bound = self.conlloovia_sol.solving_stats.lower_bound
        if window_lower_bound is None:
            if self.conlloovia_status() in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
                return self.conlloovia_cost().magnitude  # Already in usd/hour
            return np.nan
        window_size_h = self.conlloovia_sol.problem.sched_time_size.to("hour").magnitude
        return window_lower_bound / window_size_h

    def fcma_lower_bound(self, speed: int, sfmpl: float) -> float:
        """Return the lower bound of the FCMA solution at the given speed and sfmpl in
        usd/hour.
        """
        return (
            self.fcma_sols[speed, sfmpl]
            .statistics.pre_allocation_cost.to("usd/hour")
            .magnitude
        )

    def all_lower_bound_d_h(self, sfmpl: float) -> float:
        """Return the maximum between the lower bound of the obtained by Conlloovia and
        the lower bound of the FCMA speed 1 in dollars per hour."""
        c_lower_bound = self.c_lower_bound_d_h()
        fcma_lower_bound = self.fcma_lower_bound(1, sfmpl)
        if math.isnan(c_lower_bound):
            return fcma_lower_bound
        return max(c_lower_bound, fcma_lower_bound)

    def gap_cost_fcma_conlloovia_perct_str(self, speed: int, sfmpl: float) -> str:
        """Return the gap between the FCMA and Conlloovia costs as a percentage string."""
        conlloovia_cost = self.conlloovia_cost()
        fcma_cost = self.fcma_cost(speed, sfmpl)
        if conlloovia_cost.magnitude != 0:
            diff = (fcma_cost - conlloovia_cost).magnitude
            return f"{diff / conlloovia_cost.magnitude * 100:10.4f}"
        return "N/A"

    def gap_cost_fcma_c_lower_bound_perct_str(self, speed: int, sfmpl: float) -> str:
        """Return the gap between FCMA cost and Conlloovia lower bound costs as a
        percentage string."""
        fcma_cost = self.fcma_cost(speed, sfmpl)
        lower_bound = self.c_lower_bound_d_h()
        if lower_bound != np.nan and lower_bound is not None:
            return f"{(fcma_cost.magnitude - lower_bound) / lower_bound * 100:10.4f}"
        return "N/A"

    def gap_cost_fcma_all_lower_bound_perct_str(self, speed: int, sfmpl: float) -> str:
        """Return the gap between FCMA cost and the maximum lower bound costs as a
        percentage string."""
        fcma_cost = self.fcma_cost(speed, sfmpl)
        lower_bound = self.all_lower_bound_d_h(sfmpl)
        if lower_bound != np.nan and lower_bound is not None:
            return f"{(fcma_cost.magnitude - lower_bound) / lower_bound * 100:10.4f}"
        return "N/A"

    def expected_sfmpl_per_app_from_fcma(self) -> Dict[str, float]:
        """From the first FCMA solution, return a dictionary with the expected sfmpl for
        each app name. Notice that the values in FCMA are floats in rps."""
        sol = list(self.fcma_sols.values())[0]
        result = {}
        for fm in sol.allocation:
            for node in sol.allocation[fm]:
                for cgs in node.cgs:
                    app = cgs.cc.app
                    if app.name not in result:
                        result[app.name] = app.sfmpl

        return result

    def conlloovia_like_sol_fault_tolerance_m(
        self, sol: Solution, expected_sfmpl: Optional[float] = None
    ) -> float:
        """Return the fault tolerance metric of a Conlloovia-like solution .If an expected
        sfmpl is provided, use it to compute the metric. Otherwise, use the expected sfmpl
        from the FCMA solution."""
        # If a solution wasn't found, return nan
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return np.nan

        if expected_sfmpl is None:
            expected_sfmpl_per_app = self.expected_sfmpl_per_app_from_fcma()
        app_perf_data = {}
        rps_cache = {}  # ic, cc -> rps (float)
        alloc = sol.alloc

        # Obtain the sfmpl_expected, the total performance and the total performance in
        # each VM for each app
        num_warnings = 0
        for container, replicas in alloc.containers.items():
            app = container.cc.app
            vm = container.vm
            ic = vm.ic
            if replicas is None or replicas == 0:
                if num_warnings < 2:
                    print(
                        f"Warning: Replicas is None for {container.cc.name} in {vm.name()}"
                    )
                elif num_warnings == 2:
                    print("Further warnings for this solution will be suppressed")

                num_warnings += 1
                total_cc_perf_rps = 0
                continue

            if (ic, container.cc) in rps_cache:
                cc_perf_rps = rps_cache[(ic, container.cc)]
            else:
                cc_perf = sol.problem.system.perfs[ic, container.cc]
                cc_perf_rps = cc_perf.to("rps").magnitude
                rps_cache[(ic, container.cc)] = cc_perf_rps
            total_cc_perf_rps = cc_perf_rps * replicas

            if app not in app_perf_data:
                app_perf_data[app] = {
                    "sfmpl_expected": (
                        expected_sfmpl_per_app[app.name]
                        if expected_sfmpl is None
                        else expected_sfmpl
                    ),
                    "total_perf_rps": 0,
                    "vm_perfs_rps": {},
                }

            app_perf_data[app]["total_perf_rps"] += total_cc_perf_rps

            if vm not in app_perf_data[app]["vm_perfs_rps"]:
                app_perf_data[app]["vm_perfs_rps"][vm] = 0

            app_perf_data[app]["vm_perfs_rps"][vm] += total_cc_perf_rps

        # For each app, compute its sfmpl and check if it is less than or equal to the
        # expected
        n_sfmpl_apps_passed = 0  # Number of apps that fulfill the expected sfmpl
        for app, app_data in app_perf_data.items():
            # First, compute the maximum performance of any vm for this app
            max_vm_perf_rps = max(app_data["vm_perfs_rps"].values())

            # Then, compute the sfmpl for this app
            sfmpl_m = max_vm_perf_rps / app_data["total_perf_rps"]

            # Finally, check if the sfmpl_m is greater than the expected
            if sfmpl_m <= app_data["sfmpl_expected"]:
                n_sfmpl_apps_passed += 1

        # Return the proportion of apps that passed the sfmpl_m
        return n_sfmpl_apps_passed / len(app_perf_data)

    def conlloovia_like_sol_container_isolation_m(self, sol: Solution) -> float:
        """Return the container isolation metric of a Conlloovia-like solution. The metric
        is the average of 1/nc where nc is the number of containers in a VM."""
        # If a solution wasn't found, return nan
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return np.nan

        alloc = sol.alloc
        n_containers_per_vm = {}
        for container, replicas in alloc.containers.items():
            vm = container.vm
            if not alloc.vms[vm]:  # If the VM is not allocated, do not consider it
                continue
            if vm not in n_containers_per_vm:
                n_containers_per_vm[vm] = 0
            n_containers_per_vm[vm] += replicas

        isolation_metric_per_vm = [
            1 / n_containers for n_containers in n_containers_per_vm.values()
        ]
        return sum(isolation_metric_per_vm) / len(isolation_metric_per_vm)

    @staticmethod
    def compute_used_nodes(alloc: Allocation) -> dict[InstanceClass, int]:
        """Return a dictionary with the number of nodes per instance class."""
        nodes = defaultdict(int)
        for vm in alloc.vms:
            if alloc.vms[vm]:
                nodes[vm.ic] += 1

        return nodes

    @staticmethod
    def compute_common_and_total_cores(
        used_cores1: dict[InstanceClass, int], used_cores2: dict[InstanceClass, int]
    ) -> Tuple[int, int]:
        """Return the number of common cores (i.e., cores of the instance classes that are
        both in used_cores1 and in used_cores2) and total cores in used_cores1."""
        common_cores = 0
        total_cores = 0
        for ic in used_cores1:
            num_cores_ic = ic.cores.to("cores").magnitude
            common_cores += min(used_cores1[ic], used_cores2[ic]) * num_cores_ic
            total_cores += used_cores1[ic] * num_cores_ic
        return common_cores, total_cores

    def conlloovia_like_sol_vm_recycling_m(
        self, sol: Solution, sol_w2: Solution
    ) -> float:
        """Return the recycling metric of a Conlloovia-like solution. The metric is the
        proportion of cores that are reused between two allocations, i.e., the number of
        cores that are in instances that are allocated in the two allocations divided by
        the total number of cores. It takes into account that the number of cores can
        increase or decrease."""
        # If a solution wasn't found, return nan
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return np.nan

        # Same for the second window
        assert sol_w2 is not None
        if sol_w2.solving_stats.status not in (
            Status.OPTIMAL,
            Status.INTEGER_FEASIBLE,
        ):
            return np.nan

        nodes1 = ComparisonResult.compute_used_nodes(sol.alloc)
        nodes2 = ComparisonResult.compute_used_nodes(sol_w2.alloc)

        common_cores12, total_cores1 = ComparisonResult.compute_common_and_total_cores(
            nodes1, nodes2
        )
        common_cores21, total_cores2 = ComparisonResult.compute_common_and_total_cores(
            nodes2, nodes1
        )

        return max(common_cores12 / total_cores1, common_cores21 / total_cores2)

    def csol_vm_load_balance_m(self, sol: Solution) -> float:
        """Return the VM load balancing metric of a Conlloovia-like solution, which is the
        average of 1/n_a, where n_a is the number of allocated VMs for each
        application."""
        # If a solution wasn't found, return nan
        if sol.solving_stats.status not in (Status.OPTIMAL, Status.INTEGER_FEASIBLE):
            return np.nan

        alloc = sol.alloc
        vms_per_app = defaultdict(set)  # app -> set of vms
        for container in alloc.containers:
            app = container.cc.app
            vm = container.vm
            if alloc.vms[vm]:
                vms_per_app[app].add(vm)

        load_balance_metric_per_app = [1 / len(vms) for vms in vms_per_app.values()]
        return sum(load_balance_metric_per_app) / len(load_balance_metric_per_app)

    def __clear_allocations(self) -> None:
        """Remove all allocations in the solution files to free memory.

        The __setattr__ method is used to bypass the immutability of the attributes.
        """
        if self.conlloovia_sol:
            object.__setattr__(self.conlloovia_sol, "alloc", None)
        if self.conlloovia_sol_w2:
            object.__setattr__(self.conlloovia_sol_w2, "alloc", None)

        for sol in self.fcma_sols.values():
            object.__setattr__(sol, "allocation", None)
        for sol in self.fcma_sols_w2.values():
            object.__setattr__(sol, "allocation", None)

        if self.ffc_sol:
            object.__setattr__(self.ffc_sol, "alloc", None)
        if self.ffc_sol_w2:
            object.__setattr__(self.ffc_sol_w2, "alloc", None)

        if self.ffp_sol:
            object.__setattr__(self.ffp_sol, "alloc", None)
        if self.ffp_sol_w2:
            object.__setattr__(self.ffp_sol_w2, "alloc", None)

    def __compute_metrics(self) -> None:
        """Compute the metrics for the Conlloovia, FFC and FFP solutions."""
        solutions = {
            "Conlloovia": (self.conlloovia_sol, self.conlloovia_sol_w2),
            "FFC": (self.ffc_sol, self.ffc_sol_w2),
            "FFP": (self.ffp_sol, self.ffp_sol_w2),
        }

        for name, (sol, sol_w2) in solutions.items():
            self.fault_tolerance_m[name] = {
                expected_sfmpl: self.conlloovia_like_sol_fault_tolerance_m(
                    sol, expected_sfmpl
                )
                for expected_sfmpl in SFMPLS
            }
            self.container_isolation_m[name] = (
                self.conlloovia_like_sol_container_isolation_m(sol)
            )
            self.vm_recycling_m[name] = self.conlloovia_like_sol_vm_recycling_m(
                sol, sol_w2
            )
            self.vm_load_balance_m[name] = self.csol_vm_load_balance_m(sol)

    def compute_metrics_and_clear_allocs(self) -> None:
        """Compute the metrics and remove the allocations to free memory."""
        self.__compute_metrics()
        self.__clear_allocations()


class ComparisonResults:
    """Comparison results for a list of workloads. It has a list of ComparisonResult and
    provides methods to create a table and a DataFrame with the results."""

    def __init__(self, results: list[ComparisonResult]) -> None:
        self.results = results

    def append(self, result: ComparisonResult) -> None:
        """Append a new comparison result to the list."""
        self.results.append(result)

    def table(self, verbose: bool = False) -> Table:
        """Return a rich Table with the comparison results. If verbose is False, only the
        results for the first FCMA speed and SFMPL are shown."""
        if not self.results:
            raise ValueError("No results to show")

        # Get the first sfmpl and speed to create the column names in the non-verbose case
        first_sfpml = SFMPLS[0]
        first_fcma = f"FCMA_1_{first_sfpml}"

        par_names = self.results[0].par_names
        table = Table(title="Comparison Collloovia vs FCMA")
        for par_name in par_names:
            table.add_column(par_name, justify="right")
        table.add_column("Con. vars.", justify="right")
        table.add_column("Con. status")

        # Lower bounds
        table.add_column("Con. LB $/h", justify="right")
        if verbose:
            for sfmpl in SFMPLS:
                table.add_column(f"All LB sfpm={sfmpl} $/h", justify="right")
        else:
            table.add_column("All LB sfpm=1 $/h", justify="right")

        # Costs
        table.add_column("Con. $/h", justify="right")
        table.add_column("FFC cost $/h", justify="right")
        table.add_column("FFP cost $/h", justify="right")

        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(f"FCMA_{speed}_{sfmpl} $/h", justify="right")
                    table.add_column(
                        f"(FCMA_{speed}_{sfmpl}-Con.) $/h", justify="right"
                    )
                    table.add_column(
                        f"(FCMA_{speed}_{sfmpl}-Con.) gap (%)", justify="right"
                    )
                    table.add_column(
                        f"(FCMA_{speed}_{sfmpl}-C_LB) gap (%)", justify="right"
                    )
                    table.add_column(
                        f"(FCMA_{speed}_{sfmpl}-A_LB) gap (%)", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} $/h", justify="right")
            table.add_column(f"({first_fcma}-Con.) $/h", justify="right")
            table.add_column(f"({first_fcma}-Con.) gap (%)", justify="right")
            table.add_column(f"({first_fcma}-C_LB) gap (%)", justify="right")
            table.add_column(f"({first_fcma}-A_LB) gap (%)", justify="right")

        # Times
        table.add_column("Con. creation time (s)", justify="right")
        table.add_column("Con. solving time (s)", justify="right")
        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} Pre_alloc_time (s)", justify="right"
                    )
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} alloc_time (s)", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} pre_alloc_time (s)", justify="right")
            table.add_column(f"{first_fcma} alloc_time (s)", justify="right")

        # Metrics
        for sfmpl in SFMPLS:
            table.add_column(f"Con. fault_tolerance_m_e_{sfmpl}", justify="right")
        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} fault_tolerance_m", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} fault_tolerance_m", justify="right")
        table.add_column("Con. isolation_m", justify="right")
        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} isolation_m", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} isolation_m", justify="right")
        table.add_column("Con. recycling_m", justify="right")
        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} recycling_m", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} recycling_m", justify="right")
        table.add_column("Con. load_balance_m", justify="right")
        if verbose:
            for speed in SPEEDS:
                for sfmpl in SFMPLS:
                    table.add_column(
                        f"FCMA_{speed}_{sfmpl} load_balance_m", justify="right"
                    )
        else:
            table.add_column(f"{first_fcma} load_balance_m", justify="right")

        for comp_res in self.results:
            if verbose:
                all_lower_bound = (
                    f"{comp_res.all_lower_bound_d_h(sfmpl):10.4f}" for sfmpl in SFMPLS
                )
                costs_str = comp_res.costs_str()
                diff_cost_fcma_conlloovia = (
                    comp_res.diff_cost_fcma_conlloovia_str(speed, sfmpl)
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                gap_cost_fcma_conlloovia_perct = (
                    comp_res.gap_cost_fcma_conlloovia_perct_str(speed, sfmpl)
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                gap_cost_fcma_c_lower_bound_perct = (
                    comp_res.gap_cost_fcma_c_lower_bound_perct_str(speed, sfmpl)
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                gap_cost_fcma_all_lower_bound_perct = (
                    comp_res.gap_cost_fcma_all_lower_bound_perct_str(speed, sfmpl)
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )

                fcma_times = []
                for speed in SPEEDS:
                    for sfmpl in SFMPLS:
                        stats = comp_res.fcma_sols[speed, sfmpl].statistics
                        pre_alloc_time = f"{stats.pre_allocation_seconds:.4f}"
                        alloc_time = f"{stats.allocation_seconds:.4f}"
                        fcma_times.extend([pre_alloc_time, alloc_time])

                fcma_fault_tolerance_m = (
                    f"{comp_res.fcma_sols[speed, sfmpl].statistics.fault_tolerance_m:.4f}"
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                fcma_isolation_m = (
                    f"{comp_res.fcma_sols[speed, sfmpl].statistics.container_isolation_m:.4f}"
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                fcma_recycling_m = (
                    f"{comp_res.fcma_sols[speed, sfmpl].statistics.vm_recycling_m:.4f}"
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
                fcma_vm_load_balance_m = (
                    f"{comp_res.fcma_sols[speed, sfmpl].statistics.vm_load_balance_m:.4f}"
                    for speed in SPEEDS
                    for sfmpl in SFMPLS
                )
            else:
                all_lower_bound = (
                    f"{comp_res.all_lower_bound_d_h(first_sfpml):10.4f}",
                )
                costs_str = (
                    comp_res.costs_str()[0],  # Conlloovia
                    comp_res.costs_str()[1],  # FFC
                    comp_res.costs_str()[2],  # FFP
                    comp_res.costs_str()[3],  # FCMA speed 1 sfmpl 0.5
                )
                diff_cost_fcma_conlloovia = (
                    comp_res.diff_cost_fcma_conlloovia_str(1, first_sfpml),
                )
                gap_cost_fcma_conlloovia_perct = (
                    comp_res.gap_cost_fcma_conlloovia_perct_str(1, first_sfpml),
                )
                gap_cost_fcma_c_lower_bound_perct = (
                    comp_res.gap_cost_fcma_c_lower_bound_perct_str(1, first_sfpml),
                )
                gap_cost_fcma_all_lower_bound_perct = (
                    comp_res.gap_cost_fcma_all_lower_bound_perct_str(1, first_sfpml),
                )
                fcma_times = (
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.pre_allocation_seconds:.4f}",
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.allocation_seconds:.4f}",
                )
                fcma_fault_tolerance_m = (
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.fault_tolerance_m:.4f}",
                )
                fcma_isolation_m = (
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.container_isolation_m:.4f}",
                )
                fcma_recycling_m = (
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.vm_recycling_m:.4f}",
                )
                fcma_vm_load_balance_m = (
                    f"{comp_res.fcma_sols[1, first_sfpml].statistics.vm_load_balance_m:.4f}",
                )

            c_stats = comp_res.conlloovia_sol.solving_stats
            conlloovia_times = (
                f"{c_stats.creation_time:10.4f}",
                f"{c_stats.solving_time:10.4f}",
            )

            conlloovia_fault_tolerance_m = (
                str(comp_res.fault_tolerance_m["Conlloovia"][expected_sfmpl])
                for expected_sfmpl in SFMPLS
            )
            conlloovia_vm_recycling_m_number = comp_res.vm_recycling_m["Conlloovia"]
            conlloovia_vm_recycling_m = f"{conlloovia_vm_recycling_m_number:.4f}"
            conlloovia_vm_load_balance_m_number = comp_res.vm_load_balance_m[
                "Conlloovia"
            ]
            conlloovia_vm_load_balance_m = f"{conlloovia_vm_load_balance_m_number:.4f}"

            values = (str(value) for value in comp_res.par_values)
            table.add_row(
                *values,
                str(comp_res.conlloovia_vars()),
                str(comp_res.conlloovia_status().name),
                f"{comp_res.c_lower_bound_d_h():10.4f}",
                *all_lower_bound,
                *costs_str,
                *diff_cost_fcma_conlloovia,
                *gap_cost_fcma_conlloovia_perct,
                *gap_cost_fcma_c_lower_bound_perct,
                *gap_cost_fcma_all_lower_bound_perct,
                *conlloovia_times,
                *fcma_times,
                *conlloovia_fault_tolerance_m,
                *fcma_fault_tolerance_m,
                f"{comp_res.container_isolation_m["Conlloovia"]:.4f}",
                *fcma_isolation_m,
                conlloovia_vm_recycling_m,
                *fcma_recycling_m,
                conlloovia_vm_load_balance_m,
                *fcma_vm_load_balance_m,
            )

        return table

    def dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with the comparison results."""
        if not self.results:
            return pd.DataFrame()

        par_names = self.results[0].par_names
        return pd.DataFrame(self.as_dict(par_names))

    def as_dict(self, par_names):
        return {
            **{
                par_name: [comp.par_values[i] for comp in self.results]
                for i, par_name in enumerate(par_names)
            },
            "Conlloovia_vars": [comp.conlloovia_vars() for comp in self.results],
            "Conlloovia_status": [
                comp.conlloovia_status().name for comp in self.results
            ],
            "Conlloovia_lower_bound_d_h": [
                comp.c_lower_bound_d_h() for comp in self.results
            ],
            **{
                f"All_lower_bound_d_h_{sfmpl}": [
                    comp.all_lower_bound_d_h(sfmpl) for comp in self.results
                ]
                for sfmpl in SFMPLS
            },
            "Conlloovia_cost_d_h": [comp.costs()[0].magnitude for comp in self.results],
            "FFC_cost_d_h": [comp.costs()[1].magnitude for comp in self.results],
            "FFP_cost_d_h": [comp.costs()[2].magnitude for comp in self.results],
            **{
                f"Fcma_{speed}_{sfmpl}_cost_d_h": [
                    comp.fcma_cost(speed, sfmpl).magnitude for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            "Conlloovia_creation_time_s": [
                f"{comp.conlloovia_sol.solving_stats.creation_time:.4f}"
                for comp in self.results
            ],
            "Conlloovia_solving_time_s": [
                f"{comp.conlloovia_sol.solving_stats.solving_time:.4f}"
                for comp in self.results
            ],
            "FFC_creation_time_s": [
                f"{comp.ffc_sol.solving_stats.creation_time:.4f}"
                for comp in self.results
            ],
            "FFC_solving_time_s": [
                f"{comp.ffc_sol.solving_stats.solving_time:.4f}"
                for comp in self.results
            ],
            "FFP_creation_time_s": [
                f"{comp.ffp_sol.solving_stats.creation_time:.4f}"
                for comp in self.results
            ],
            "FFP_solving_time_s": [
                f"{comp.ffp_sol.solving_stats.solving_time:.4f}"
                for comp in self.results
            ],
            **{
                f"Fcma_{speed}_{sfmpl}_pre_alloc_time_s": [
                    f"{comp.fcma_sols[speed, sfmpl].statistics.pre_allocation_seconds:.4f}"
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            **{
                f"Fcma_{speed}_{sfmpl}_alloc_time_s": [
                    f"{comp.fcma_sols[speed, sfmpl].statistics.allocation_seconds:.4f}"
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            **{
                f"Conlloovia_fault_tolerance_m_e_{expected_sfmpl}": [
                    comp.fault_tolerance_m["Conlloovia"][expected_sfmpl]
                    for comp in self.results
                ]
                for expected_sfmpl in SFMPLS
            },
            **{
                f"FFC_fault_tolerance_m_e_{expected_sfmpl}": [
                    comp.fault_tolerance_m["FFC"][expected_sfmpl]
                    for comp in self.results
                ]
                for expected_sfmpl in SFMPLS
            },
            **{
                f"FFP_fault_tolerance_m_e_{expected_sfmpl}": [
                    comp.fault_tolerance_m["FFP"][expected_sfmpl]
                    for comp in self.results
                ]
                for expected_sfmpl in SFMPLS
            },
            **{
                f"Fcma_{speed}_{sfmpl}_fault_tolerance_m": [
                    comp.fcma_sols[speed, sfmpl].statistics.fault_tolerance_m
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            "Conlloovia_isolation_m": [
                comp.container_isolation_m["Conlloovia"] for comp in self.results
            ],
            "FFC_isolation_m": [
                comp.container_isolation_m["FFC"] for comp in self.results
            ],
            "FFP_isolation_m": [
                comp.container_isolation_m["FFP"] for comp in self.results
            ],
            **{
                f"Fcma_{speed}_{sfmpl}_isolation_m": [
                    comp.fcma_sols[speed, sfmpl].statistics.container_isolation_m
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            "Conlloovia_vm_recycling_m": [
                comp.vm_recycling_m["Conlloovia"] for comp in self.results
            ],
            "FFC_vm_recycling_m": [comp.vm_recycling_m["FFC"] for comp in self.results],
            "FFP_vm_recycling_m": [comp.vm_recycling_m["FFP"] for comp in self.results],
            **{
                f"Fcma_{speed}_{sfmpl}_vm_recycling_m": [
                    comp.fcma_sols[speed, sfmpl].statistics.vm_recycling_m
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
            "Conlloovia_vm_load_balance_m": [
                comp.vm_load_balance_m["Conlloovia"] for comp in self.results
            ],
            "FFC_vm_load_balance_m": [
                comp.vm_load_balance_m["FFC"] for comp in self.results
            ],
            "FFP_vm_load_balance_m": [
                comp.vm_load_balance_m["FFP"] for comp in self.results
            ],
            **{
                f"Fcma_{speed}_{sfmpl}_vm_load_balance_m": [
                    comp.fcma_sols[speed, sfmpl].statistics.vm_load_balance_m
                    for comp in self.results
                ]
                for speed in SPEEDS
                for sfmpl in SFMPLS
            },
        }

    def save_to_csv(
        self, filename: None | str = None, out_dir: None | str = None
    ) -> None:
        """Save the comparison results to a CSV file."""
        if filename is None:
            date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fcma_conlloovia_comparison_{date_str}.csv"
        if out_dir is not None:
            filename = f"{out_dir}/{filename}"

        # Check that the out_dir exists and create it if it doesn't
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.dataframe().to_csv(filename, index=False)

    def save_to_pickle(
        self, filename: None | str = None, out_dir: None | str = None
    ) -> None:
        """Save the results to a pickle file."""
        if filename is None:
            date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fcma_conlloovia_solutions_{date_str}.p"
        if out_dir is not None:
            filename = f"{out_dir}/{filename}"

        # Check that the out_dir exists and create it if it doesn't
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as file:
            pickle.dump(self.results, file)


class SolverParams:
    """Parameters for the CBC solvers."""

    def __init__(
        self,
        frac_gap: None | float,
        max_seconds: None | int,
        threads: int,
        seed: int,
    ) -> None:
        self.frac_gap = frac_gap
        self.max_seconds = max_seconds
        self.threads = threads
        self.seed = seed


def create_problem_with_updated_sfmpl(
    fcma_problem: fcma.Fcma, sfmpl: float
) -> fcma.Fcma:
    """Change the sfmpl of all the applications of the FCMA problem and return it as a new
    problem."""
    new_system = {}
    new_apps = {}  # old_app -> new_app
    for old_app, fm in fcma_problem.system.keys():
        new_app = fcma.App(name=old_app.name, sfmpl=sfmpl)
        new_apps[old_app] = new_app
        new_system[new_app, fm] = fcma_problem.system[old_app, fm]

    new_workloads = {
        new_apps[app]: reqs for app, reqs in fcma_problem.workloads.items()
    }

    return fcma.Fcma(system=new_system, workloads=new_workloads)


class ComparatorFcmaConllooviaFirstFit:
    """Compares a problem between FCMA, Conlloovia and the first fit heuristics FFP and
    FFC."""

    def __init__(
        self,
        fcma_problem: fcma.Fcma,
        conlloovia_problem: Problem,
        par_names: list[str],
        par_values: list[float],
        dir_name: str,
    ) -> None:
        self.fcma_problem = fcma_problem
        self.conlloovia_problem = conlloovia_problem
        self.par_names = par_names
        self.par_values = par_values
        self.dir_name = dir_name

        self.log = logging.getLogger("rich")

    def dump_sol(
        self, sol: Solution, filename_prefix: str, filename_suffix: str
    ) -> None:
        """Save the solution to a pickle file."""
        params_str = "_".join(str(param) for param in self.par_values)
        filename = (
            f"{self.dir_name}/{filename_prefix}sol_{params_str}{filename_suffix}.p"
        )

        with open(filename, "wb") as file:
            pickle.dump(sol, file)

    def compare(
        self, solver_params: SolverParams, verbose: bool = False
    ) -> ComparisonResult:
        """Compare the FCMA and Conlloovia solutions."""
        self.log.info(
            f"Comparing with pars: {list(zip(self.par_names, self.par_values))}"
        )

        solver = PULP_CBC_CMD(
            gapRel=solver_params.frac_gap,
            threads=solver_params.threads,
            timeLimit=solver_params.max_seconds,
            msg=verbose,
            options=[f"randomS {solver_params.seed} randomC {solver_params.seed}"],
        )

        # All problems are solved twice, the second time with the workload multiplied by
        # this value. This is done to check the recycling metric
        wl_multiplier = 1.2

        fcma_sols, fcma_sols_w2 = self.solve_fcma_2_windows(solver, wl_multiplier)

        conlloovia_sol, conlloovia_sol_w2 = self.solve_conlloovia_2_windows(
            solver_params, verbose, solver, wl_multiplier
        )

        ffc_sol, ffc_sol_w2 = self.solve_first_fist_2_windows(
            FirstFitIcOrdering.CORE_DESCENDING, wl_multiplier
        )
        ffp_sol, ffp_sol_w2 = self.solve_first_fist_2_windows(
            FirstFitIcOrdering.PRICE_ASCENDING, wl_multiplier
        )

        return ComparisonResult(
            conlloovia_sol,
            conlloovia_sol_w2,
            fcma_sols,
            fcma_sols_w2,
            ffc_sol,
            ffc_sol_w2,
            ffp_sol,
            ffp_sol_w2,
            self.par_names,
            self.par_values,
        )

    def solve_fcma_2_windows(
        self, solver: Any, wl_multiplier: float
    ) -> tuple[FcmaSolutionDict, FcmaSolutionDict]:
        """Solve the FCMA problem for the first and second windows."""
        fcma_sols: FcmaSolutionDict = {}  # [speed, sfmpl] -> Solution
        fcma_sols_w2: FcmaSolutionDict = {}  # [speed, sfmpl] -> Solution
        for speed in SPEEDS:
            for sfmpl in SFMPLS:
                self.log.info(f"Solving FCMA {speed} with sfmpl {sfmpl}")
                if sfmpl != 1:
                    fcma_problem_to_solve = create_problem_with_updated_sfmpl(
                        self.fcma_problem, sfmpl
                    )
                else:
                    fcma_problem_to_solve = self.fcma_problem
                fcma_sols[speed, sfmpl] = fcma_problem_to_solve.solve(
                    fcma.SolvingPars(
                        speed_level=speed,
                        solver=solver,
                    )
                )
                self.dump_sol(fcma_sols[speed, sfmpl], "fcma_", f"_{speed}_{sfmpl}")

                # Solve the second window
                self.log.info(
                    f"Solving FCMA {speed} with sfmpl {sfmpl} in the second window"
                )
                fcma_wl_w2 = {
                    key: value * wl_multiplier
                    for key, value in fcma_problem_to_solve.workloads.items()
                }
                fcma_problem_w2 = fcma.Fcma(fcma_problem_to_solve.system, fcma_wl_w2)
                fcma_sols_w2[speed, sfmpl] = fcma_problem_w2.solve(
                    fcma.SolvingPars(
                        speed_level=speed,
                        solver=solver,
                    )
                )

                fcma_sols[speed, sfmpl].statistics.update_metrics(
                    fcma_sols[speed, sfmpl].allocation,
                    fcma_sols_w2[speed, sfmpl].allocation,
                )

        return fcma_sols, fcma_sols_w2

    def solve_conlloovia_2_windows(
        self,
        solver_params: SolverParams,
        verbose: bool,
        solver: Any,
        wl_multiplier: float,
    ) -> tuple[Solution, Solution]:
        """Solve the Conlloovia problem for the first and second windows."""
        self.log.info("Solving Conlloovia")
        conlloovia_solution = self.solve_conlloovia(self.conlloovia_problem, solver)
        self.dump_sol(conlloovia_solution, "conlloovia_", "")

        # Sometimes, the solver says the solution is not feasible but there is a solution.
        # It's a bug when using presolving and preprocessing. So if the solution obtained
        # is not feasible, we try to solve it without presolving
        if conlloovia_solution.solving_stats.status == Status.INFEASIBLE:
            self.log.info(
                "Infeasible solution found. Trying to solve Conlloovia without presolving"
            )
            solver_no_pre = PULP_CBC_CMD(
                gapRel=solver_params.frac_gap,
                threads=solver_params.threads,
                timeLimit=solver_params.max_seconds,
                msg=verbose,
                options=[
                    f"randomS {solver_params.seed} randomC {solver_params.seed} "
                    f"preprocess off presolve off"
                ],
            )
            conlloovia_solution = self.solve_conlloovia(
                self.conlloovia_problem, solver_no_pre
            )
            self.dump_sol(conlloovia_solution, "conlloovia_", "_no_pre")

        # Solve Conlloovia for the second window if there is an optimal or feasible
        # solution for the first window
        conlloovia_solution_w2 = None
        if conlloovia_solution.solving_stats.status in (
            Status.OPTIMAL,
            Status.INTEGER_FEASIBLE,
        ):
            self.log.info("Solving Conlloovia in the second window")
            conlloovia_problem_w2 = Problem(
                system=self.conlloovia_problem.system,
                workloads={
                    app: Workload(
                        num_reqs=wl.num_reqs * wl_multiplier,
                        time_slot_size=wl.time_slot_size,
                        app=app,
                    )
                    for app, wl in self.conlloovia_problem.workloads.items()
                },
                sched_time_size=self.conlloovia_problem.sched_time_size,
            )
            conlloovia_solution_w2 = self.solve_conlloovia(
                conlloovia_problem_w2, solver
            )
        else:
            self.log.info(
                "There is no feasible solution for the first window, so we don't solve "
                "the second window"
            )

        return conlloovia_solution, conlloovia_solution_w2

    def solve_first_fist_2_windows(
        self, ordering: FirstFitIcOrdering, wl_multiplier: float
    ) -> tuple[Solution, Solution]:
        self.log.info(f"Solving First Fit: {ordering}")
        ff_solution = self.solve_first_fit(self.conlloovia_problem, ordering)
        prefix = "ffc_" if ordering == FirstFitIcOrdering.CORE_DESCENDING else "ffp_"
        self.dump_sol(ff_solution, prefix, "")

        # Solve for the second window if there is an optimal or feasible solution for the
        # first window
        if ff_solution.solving_stats.status in (
            Status.OPTIMAL,
            Status.INTEGER_FEASIBLE,
        ):
            self.log.info(f"Solving First Fit in the second window: {ordering}")
            ff_problem_w2 = Problem(
                system=self.conlloovia_problem.system,
                workloads={
                    app: Workload(
                        num_reqs=wl.num_reqs * wl_multiplier,
                        time_slot_size=wl.time_slot_size,
                        app=app,
                    )
                    for app, wl in self.conlloovia_problem.workloads.items()
                },
                sched_time_size=self.conlloovia_problem.sched_time_size,
            )
            ff_solution_w2 = self.solve_first_fit(ff_problem_w2, ordering)

        return ff_solution, ff_solution_w2

    def solve_conlloovia(self, problem: Problem, solver: Any) -> Solution:
        """Solve the Conlloovia problem."""
        self.log.info("Adapting the problem")
        adapted_problem = LimitsAdapter(problem=problem).compute_adapted_problem()
        self.log.info("Allocating with Conlloovia")
        conlloovia_alloc = ConllooviaAllocator(adapted_problem)
        return conlloovia_alloc.solve(solver)

    def solve_first_fit(
        self, problem: Problem, ordering: FirstFitIcOrdering
    ) -> Solution:
        """Solve the Conlloovia problem with the first fit heuristics."""
        self.log.info("Adapting the problem")
        adapted_problem = LimitsAdapter(problem=problem).compute_adapted_problem()
        self.log.info(f"Allocating with First Fit {ordering}")
        first_fit_alloc = FirstFitAllocator(adapted_problem, ordering)
        return first_fit_alloc.solve()


class Fcma2Conlloovia:
    """Convert an FCMA problem to a Conlloovia problem."""

    def __init__(self, fcma_problem: fcma.Fcma) -> None:
        self.fcma_problem = fcma_problem

    def convert(self, add_extra_ccs: bool) -> Problem:
        """Convert the FCMA problem to a Conlloovia problem. The time units will be 1
        hour.

        Parameters:
        - add_extra_ccs: If True, add two extra container classes to each app family, one
          with the minimum number of cores multiplied by 1.5 and the other with the
          maximum number of cores multiplied by 1.5.
        """
        # Conlloovia apps
        f2c_app = {
            f_app: App(name=f_app.name)
            for f_app, _ in self.fcma_problem.workloads.items()
        }
        c_apps = tuple(f2c_app.values())

        # Conlloovia instance classes
        keys = self.fcma_problem.system.keys()  # Each key is a tuple (app, family)
        f_fms = tuple(fm for _, fm in keys)
        f2c_ic = {}  # Map from FCMA instance classes to Conlloovia instance classes
        for fm in f_fms:
            for f_ic in fm.ics:
                if f_ic not in f2c_ic:
                    f2c_ic[f_ic] = InstanceClass(
                        name=f_ic.name,
                        price=f_ic.price,
                        cores=f_ic.cores,
                        mem=f_ic.mem,
                        limit=1,  # It should be adapted later
                    )
        c_ics = tuple(f2c_ic.values())

        # Conlloovia container classes and perfs
        c_ccs_list = []
        c_perfs = {}
        for (f_app, fm), perf in self.fcma_problem.system.items():
            # Each FCMA family has a set of aggregations of containers, which will be
            # represented as different container classes in Conlloovia. In addition, if
            # add_extra_ccs is true, two extra aggregations are added: min*1.5 and max*1.5
            aggs = perf.aggs
            if add_extra_ccs:
                aggs += (1.5 * perf.aggs[0], 1.5 * perf.aggs[-1])

            # This converter only works if the memory is constant for all the
            # aggregations, so let's check
            if any(p != perf.mem[0] for p in perf.mem):
                raise ValueError("Memory should be constant for all the aggregations")
            mem = perf.mem[0]

            for agg in aggs:
                c_cc = ContainerClass(
                    name=f"{f_app.name}_{fm.name}_{agg}",
                    cores=perf.cores * agg,
                    mem=mem,
                    app=f2c_app[f_app],
                )
                c_ccs_list.append(c_cc)

                for f_ic in fm.ics:
                    c_ic = f2c_ic[f_ic]
                    c_perfs[(c_ic, c_cc)] = perf.perf * agg

        c_ccs = tuple(c_ccs_list)

        # Conlloovia workloads
        c_workloads = {}
        for f_app, reqs in self.fcma_problem.workloads.items():
            reqs_window = Requests(f"{reqs.magnitude} reqs")
            c_workloads[f2c_app[f_app]] = Workload(
                num_reqs=reqs_window, time_slot_size=Time("1 h"), app=f2c_app[f_app]
            )

        system = System(apps=c_apps, ics=c_ics, ccs=c_ccs, perfs=c_perfs)

        return Problem(
            system=system, workloads=c_workloads, sched_time_size=Time("1 h")
        )
