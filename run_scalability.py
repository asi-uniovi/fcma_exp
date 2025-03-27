"""This module carries out scalability experiments for FCMA."""

import datetime

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import shutil
from typing import List, Optional

import numpy
from rich import print  # pylint: disable=redefined-builtin
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table

from pulp import PULP_CBC_CMD  # type: ignore

import fcma
from fcma.serialization import ProblemSerializer
from fcma.model import SolutionSummary

from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    RequestsPerTime,
    Storage,
)

import aws_eu_west_1

FRAC_GAP = 0.02
MAX_SECONDS = 30
SEED = 150
COST_LIMIT = CurrencyPerTime(
    "100 usd/hour"
)  # if the minimum cost is higher than this, the experiment is discarded
MAX_APPS = 100  # Very big, so that the real limit is the cost limit
MAX_EXPS = 1000


@dataclass
class ExperimentInfo:
    num: List[int] = field(default_factory=list)
    apps: List[int] = field(default_factory=list)
    num_families: List[int] = field(default_factory=list)
    total_secs: List[Optional[float]] = field(default_factory=list)
    pre_allocation_secs: List[Optional[float]] = field(
        default_factory=list
    )  # Included in pre_allocation_secs
    partial_ilp_seconds: List[Optional[float]] = field(default_factory=list)
    allocation_secs: List[Optional[float]] = field(default_factory=list)
    pre_allocation_lower_bound_cost: List[Optional[float]] = field(
        default_factory=list
    )
    cost: List[Optional[float]] = field(default_factory=list)
    min_pred_cost: List[Optional[float]] = field(default_factory=list)
    num_vms: List[Optional[int]] = field(default_factory=list)
    failed: List[bool] = field(default_factory=list)
    timed_out: List[bool] = field(default_factory=list)
    problem_file: List[str] = field(default_factory=list)
    solution_file: List[str] = field(default_factory=list)

    def save_experiment_data(
        self,
        num_exps: int,
        num_apps: int,
        families: list,
        sol: fcma.Solution,
        min_pred_cost: float,
        vm_summary: dict,
        problem_file: str,
        solution_file: str,
    ):
        """Save the information of the experiment."""
        self.num.append(num_exps)
        self.apps.append(num_apps)
        self.num_families.append(len(families))
        self.total_secs.append(
            sol.statistics.total_seconds
            if sol.statistics.final_cost is not None
            else None
        )
        self.pre_allocation_secs.append(
            sol.statistics.pre_allocation_seconds
            if sol.statistics.final_cost is not None
            else None
        )
        self.partial_ilp_seconds.append(
            sol.statistics.partial_ilp_seconds
            if sol.statistics.final_cost is not None
            else None
        )
        self.allocation_secs.append(
            sol.statistics.allocation_seconds
            if sol.statistics.final_cost is not None
            else None
        )
        self.pre_allocation_lower_bound_cost.append(
            sol.statistics.pre_allocation_lower_bound_cost.magnitude
            if sol.statistics.pre_allocation_lower_bound_cost is not None
            else None
        )
        self.cost.append(
            sol.statistics.final_cost.magnitude
            if sol.statistics.final_cost is not None
            else None
        )
        self.min_pred_cost.append(min_pred_cost.to("usd/hour").magnitude)
        self.num_vms.append(
            vm_summary.total_num if sol.statistics.final_cost is not None else None
        )
        self.problem_file.append(problem_file)
        self.solution_file.append(solution_file)

    def display_experiment_info(
        self,
        num_exps: int,
        failed_exps: int,
        timeout_exps: int,
        discarded_exps: int,
    ):
        """Display the experiment information in a rich table."""
        table = Table(title="Scaling Experiment")
        table.add_column("Num", style="cyan")
        table.add_column("Apps", style="magenta")
        table.add_column("Num Families", style="yellow")
        table.add_column("Pre Allocation Secs", style="magenta")
        table.add_column("Partial ILP Secs", style="magenta")
        table.add_column("Allocation Secs", style="magenta")
        table.add_column("Total Secs", style="magenta")
        table.add_column("Bound ($/h)", style="red")
        table.add_column("Cost ($/h)", style="red")
        table.add_column("Min Pred Cost ($/h)", style="red")
        table.add_column("Num VMs", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Timed Out", style="red")
        for j in range(num_exps + 1):
            table.add_row(
                str(self.num[j]),
                str(self.apps[j]),
                str(self.num_families[j]),
                (
                    f"{self.pre_allocation_secs[j]:.4f}"
                    if self.pre_allocation_secs[j] is not None
                    else "-"
                ),
                (
                    f"{self.partial_ilp_seconds[j]:.4f}"
                    if self.partial_ilp_seconds[j] is not None
                    else "-"
                ),
                (
                    f"{self.allocation_secs[j]:.4f}"
                    if self.allocation_secs[j] is not None
                    else "-"
                ),
                f"{self.total_secs[j]:.4f}" if self.total_secs[j] is not None else "-",
                (
                    f"{self.pre_allocation_lower_bound_cost[j]:.4f}"
                    if self.pre_allocation_lower_bound_cost[j] is not None
                    else "-"
                ),
                f"{self.cost[j]:.4f}" if self.cost[j] is not None else "-",
                (
                    f"{self.min_pred_cost[j]:.4f}"
                    if self.min_pred_cost[j] is not None
                    else "-"
                ),
                str(self.num_vms[j]),
                str(self.failed[j]),
                str(self.timed_out[j]),
            )
        print(table)
        print(f"Number of experiments: {num_exps + 1}")
        print(
            f"Number of failed experiments: {failed_exps} ({failed_exps / (num_exps + 1) * 100:.2f}%)"
        )
        print(
            f"Number of timed out experiments: {timeout_exps} ({timeout_exps / (num_exps + 1) * 100:.2f}%)"
        )
        print(
            f"Number of discarded experiments: {discarded_exps} ({discarded_exps / (discarded_exps + num_exps + 1) * 100:.2f}%)"
        )

    def save_csv(self, dir_name: str, num_exps: int):
        """Save the information of the experiments to a csv file."""
        with open(f"{dir_name}/exp_info.csv", "w") as file:
            fields = [
                "num",
                "apps",
                "num_families",
                "total_secs",
                "pre_allocation_secs",
                "allocation_secs",
                "partial_ilp_seconds",
                "pre_allocation_lower_bound_cost",
                "cost",
                "min_pred_cost",
                "num_vms",
                "failed",
                "timed_out",
                "problem_file",
                "solution_file",
            ]
            file.write(",".join(fields))
            file.write("\n")
            for j in range(num_exps):
                row = [str(getattr(self, exp_field)[j]) for exp_field in fields]
                file.write(",".join(row))
                file.write("\n")

        # Copy the csv file to the current directory so that that it has the latest
        # information
        src = os.path.join(dir_name, "exp_info.csv")
        dst = os.path.join(".", "data_scaling.csv")
        shutil.copy(src, dst)

def solve(problem: fcma.Fcma, verbose: bool = False) -> fcma.Solution:
    """Solve the problem using the PULP solver."""
    solver = PULP_CBC_CMD(
        gapRel=FRAC_GAP,
        timeLimit=MAX_SECONDS,
        msg=verbose,
        options=[f"randomS {SEED} randomC {SEED}"],
    )
    return problem.solve(fcma.SolvingPars(speed_level=1, solver=solver))


def compute_min_cost_app(app: fcma.App, problem: fcma.Fcma) -> float:
    """Compute the minimum cost for a given app.

    The cost is computed as the minimum cost of the instance class with the minimum price
    per core in the family. The minimum cost of the instance is computed as the number of
    cores needed to satisfy the workload of the app divided by the performance per core of
    that instance class.
    """
    min_cost = None
    families = {family for _, family in problem.system.keys()}
    for family in families:
        if (app, family) not in problem.system:
            continue

        fam_cores = problem.system[(app, family)].cores
        fam_perf = problem.system[(app, family)].perf
        fam_perf_per_core = fam_perf / fam_cores
        app_workload = problem.workloads[app]
        req_cores = app_workload / fam_perf_per_core

        # Loop over all the container classes for this family
        cheapest_ic_per_core = None
        best_cost_per_core = float("inf")
        for ic in family.ics:
            cost_per_core = ic.price / ic.cores
            if cheapest_ic_per_core is None or cost_per_core < best_cost_per_core:
                cheapest_ic_per_core = ic
                best_cost_per_core = cost_per_core

        # Notice that this is an approximation because it considers that we only pay the
        # cores we need, but we are actually paying the whole instance.
        cost_family = req_cores * best_cost_per_core

        if min_cost is None or cost_family < min_cost:
            min_cost = cost_family

    if min_cost is None:
        raise ValueError("No family found for app")

    return min_cost


def compute_min_cost(problem: fcma.Fcma) -> float:
    """Compute the minimum cost of the problem.

    The cost is computed as the sum of the minimum cost of each app. This is a lower
    bound.
    """
    cost_min_total = 0
    apps = {app for app, _ in set(problem.system.keys())}
    for app in apps:
        c_app = compute_min_cost_app(app, problem)
        cost_min_total += c_app

    return cost_min_total

def compute_price_per_core_per_fm(
    families: tuple[fcma.InstanceClassFamily],
) -> dict[fcma.InstanceClassFamily, float]:
    """Returns a dictionary with the price per core for each family. The price per core is
    computed as the price of the instance class with the minimum price per core in the
    family."""
    price_per_core_per_fm = {}
    for fm in families:
        price_per_core_per_fm[fm] = compute_min_ic_price_per_core(fm)
    return price_per_core_per_fm


def compute_min_ic_price_per_core(fm: fcma.InstanceClassFamily) -> float:
    """Compute the price per core of the instance class with the minimum price per core in
    the family."""
    min_ic_price_per_core = float("inf")
    for ic in fm.ics:
        ic_price_per_core = ic.price.to("usd/hour").magnitude / ic.cores.magnitude
        min_ic_price_per_core = min(min_ic_price_per_core, ic_price_per_core)

    return min_ic_price_per_core

def create_fcma_problem(
    num_apps: int,
    families: tuple[fcma.InstanceClassFamily],
) -> fcma.Fcma:
    """Create a synthetic FCMA problem.

    Args:
        num_apps: Number of apps.
        families: Instance class families.
    """
    apps = tuple(fcma.App(name=f"app_{i}") for i in range(num_apps))

    # The workloads will be always 1 rps and the performance is what changes
    workloads = {app: RequestsPerTime("1 req/s") for app in apps}

    # Compute for each family the instance class with minimum number of GiB per core
    price_per_core_per_fm = compute_price_per_core_per_fm(families)

    min_price_per_core_all_fms = min(price_per_core_per_fm.values())

    system_dict = {}
    for fm in families:
        # Compute the relation betwen the price_per_core of the family and the
        # min_price_per_core_all_fms
        price_per_core = price_per_core_per_fm[fm]

        # 5 values around "price_per_core_relation"
        mul_price_opts = [0.75, 0.9, 1, 1.15, 1.2]
        mul_price = random.choice(mul_price_opts)
        price_per_core_relation = (
            price_per_core / min_price_per_core_all_fms
        ) * mul_price

        for app in apps:
            base_cores = random.uniform(0.1, 2) # Average around 1 core
            mul_cores_opts = [0.75, 0.9, 1, 1.15, 1.2]  # 5 core sizes around "base_cores"
            mul_cores = random.choice(mul_cores_opts)
            actual_cores_app = base_cores * mul_cores * price_per_core_relation

            mem_cores_rel = random.choice([2, 4, 8, 16])
            mul_mem_opts = [0.75, 0.9, 1, 1.15, 1.2]  # 5 values around "mem_cores_rel"
            mul_mem = random.choice(mul_mem_opts)
            mem = actual_cores_app * mem_cores_rel * mul_mem

            perf = random.uniform(0.01, 1)
            mul_perf_opts = [0.04, 0.08, 0.16, 0.32, 0.4, 0.8, 1.1, 2.1, 4]
            mul_perf = random.choice(mul_perf_opts)
            actual_perf = perf * mul_perf * price_per_core_relation

            # The aggregations can be 2, 4 or 8, but never over 1 core
            aggs = tuple(agg for agg in (2, 4, 8) if agg * actual_cores_app < 1)

            system_dict[(app, fm)] = fcma.AppFamilyPerf(
                cores=ComputationalUnits(f"{actual_cores_app} cores"),
                mem=Storage(f"{mem} gibibytes"),
                perf=RequestsPerTime(f"{actual_perf} req/s"),
                aggs=aggs,
            )

    return fcma.Fcma(fcma.System(system_dict), workloads=workloads)

def main() -> None:
    """Main function for the scaling experiments."""
    random.seed(100)  # For reproducibility
    numpy.random.seed(100)  # For reproducibility

    # For saving the intermediary files
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"out_scaling/{date_str}"

    # Create an output directory for intermediary files
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    possible_families = [
        aws_eu_west_1.c5_m5_r5_fm,
        aws_eu_west_1.c6i_m6i_r6i_fm,
        aws_eu_west_1.c6g_m6g_r6g_fm,
        aws_eu_west_1.c7a_m7a_r7a_fm,
    ]

    # Create a dict to save the information of the experiments
    exp_info = ExperimentInfo()

    failed_exps = 0
    timeout_exps = 0
    discarded_exps = 0
    num_exp = 0
    with Progress(
        *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Experiments", total=MAX_EXPS)
        while num_exp < MAX_EXPS:
            num_apps = random.randint(1, MAX_APPS)
            num_families = random.randint(1, len(possible_families))
            print(f"Experiment {num_exp}: {num_apps} apps, {num_families} families")
            families = random.sample(possible_families, num_families)

            problem = create_fcma_problem(num_apps=num_apps, families=families)

            min_pred_cost = compute_min_cost(problem)
            print(f"  Experiment {num_exp}: {num_apps} apps, {len(families)} "
                  f"families, min cost: {min_pred_cost}")
            if min_pred_cost > COST_LIMIT:
                print(f"    Problem too expensive ({min_pred_cost} > {COST_LIMIT})")
                discarded_exps += 1
                continue  # Skip the experiment

            exp_desc = f"e_{num_exp}_a_{num_apps}_f_{len(families)}"
            problem_file = f"{dir_name}/problem_{exp_desc}.json"
            with open(problem_file, "w", encoding="utf-8") as file:
                problem_serializer = ProblemSerializer(problem)
                problem_json = problem_serializer.as_dict()
                json.dump(problem_json, file, indent=4)

            sol = solve(problem)

            solution_file = ""
            if sol.statistics.final_cost is None:
                print("     The problem could not be solved")
                print(f"    Problem file: {problem_file}")

                # Print information about the problem and the solution
                failed_exps += 1
                exp_info.failed.append(True)
            else:
                summary = SolutionSummary(sol)
                vm_summary = summary.get_vm_summary()
                print(vm_summary)

                exp_info.failed.append(False)

                solution_file = f"{dir_name}/sol_{exp_desc}.json"
                with open(solution_file, "w", encoding="utf-8") as file:
                    ss = SolutionSummary(sol)
                    file.write(json.dumps(ss.as_dict(), indent=2))

            if (
                sol.statistics.total_seconds is None
                or sol.statistics.total_seconds > MAX_SECONDS
            ):
                timeout_exps += 1
                exp_info.timed_out.append(True)
            else:
                exp_info.timed_out.append(False)

            exp_info.save_experiment_data(
                num_exp,
                num_apps,
                families,
                sol,
                min_pred_cost,
                vm_summary,
                problem_file,
                solution_file,
            )
            exp_info.display_experiment_info(
                num_exp, failed_exps, timeout_exps, discarded_exps
            )
            exp_info.save_csv(dir_name=dir_name, num_exps=num_exp+1)

            num_exp += 1
            progress.update(task, advance=1)


if __name__ == "__main__":
    main()
