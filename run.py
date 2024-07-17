"""This module compares FCMA and Conlloovia using synthetic data."""

import datetime
import itertools
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Optional

import numpy
from rich import print  # pylint: disable=redefined-builtin
from rich.progress import track

import fcma
from fcma.serialization import ProblemSerializer

from cloudmodel.unified.units import (
    ComputationalUnits,
    RequestsPerTime,
    Storage,
)

sys.path.insert(0, "/home/joaquin/k8s/fcma")  # pylint: disable=wrong-import-position
from examples import aws_eu_west_1

from comparator import (
    Fcma2Conlloovia,
    ComparatorFcmaConlloovia,
    ComparisonResult,
    ComparisonResults,
    SolverParams,
)


def create_fcma_problem(
    num_apps: int,
    families: tuple[fcma.InstanceClassFamily],
    base_cores: float,
    mem_cores_rel: int,
    perf: float,
) -> fcma.Fcma:
    """Create a synthetic FCMA problem.

    Args:
        num_apps: Number of apps.
        families: Instance class families.
        base_cores: Number of cores for the minimum container. Must be 0.120 or 3. The
            actual number of cores will be a value around this number.
        mem_cores_rel: Relation between memory and cores. The actual memory will be around
            actual_cores_app * mem_mul.
        perf: Performance of the minimum container class in the family with the minimum
            price per core. The actual performance will be a value around this and
            proportional to the price.
    """
    apps = tuple(fcma.App(name=f"app_{i}") for i in range(num_apps))

    # The workloads will be always 1 rps and the performance is what changes
    workloads = {app: RequestsPerTime("1 req/s") for app in apps}

    if base_cores not in (0.120, 3):
        raise ValueError("Invalid number of cores. Must be 0.120 or 3")

    # Compute for each family the instance class with minimum number of GiB per core
    price_per_core_per_fm = compute_price_per_core_per_fm(families)

    min_price_per_core_all_fms = min(price_per_core_per_fm.values())
    print(f"min_price_per_core_all_fms = {min_price_per_core_all_fms}")

    system_dict = {}
    for fm in families:
        # Compute the relation betwen the price_per_core of the family and the
        # min_price_per_core_all_fms
        price_per_core = price_per_core_per_fm[fm]

        mul_price_opts = [
            0.75,
            0.9,
            1,
            1.15,
            1.2,
        ]  # 5 values around "price_per_core_relation"
        mul_price = random.choice(mul_price_opts)
        price_per_core_relation = (
            price_per_core / min_price_per_core_all_fms
        ) * mul_price

        for app in apps:
            mul_cores_opts = [0.75, 0.9, 1, 1.15, 1.2]  # 5 core sizes around "cores"
            mul_cores = random.choice(mul_cores_opts)
            actual_cores_app = base_cores * mul_cores * price_per_core_relation

            mul_mem_opts = [0.75, 0.9, 1, 1.15, 1.2]  # 5 values around "mem_cores_rel"
            mul_mem = random.choice(mul_mem_opts)
            mem = actual_cores_app * mem_cores_rel * mul_mem

            mul_perf_opts = [0.04, 0.08, 0.16, 0.32, 0.4, 0.8, 1.1, 2.1, 4]
            mul_perf = random.choice(mul_perf_opts)
            actual_perf = perf * mul_perf * price_per_core_relation

            # The aggregation for apps with small containers are easier because they are
            # not multithreaded
            aggs = (2, 4, 8) if base_cores == 0.120 else (2,)

            system_dict[(app, fm)] = fcma.AppFamilyPerf(
                cores=ComputationalUnits(f"{actual_cores_app} cores"),
                mem=Storage(f"{mem} gibibytes"),
                perf=RequestsPerTime(f"{actual_perf} req/s"),
                aggs=aggs,
            )

            # Print logging information about the multipliers used
            print(
                f"app: {app.name} fm: {fm.name}\n"
                f"   cores: {actual_cores_app} mem: {mem} perf: {actual_perf}\n"
                f"   mul_cores: {mul_cores} mul_mem: {mul_mem} mul_perf: {mul_perf}"
            )
    return fcma.Fcma(fcma.System(system_dict), workloads=workloads)


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


def compare_scenario(
    exp_num: int,
    napps: int,
    families: tuple[fcma.InstanceClassFamily],
    base_cores: float,
    mem_mul: int,
    perf: float,
    frac_gap: float,
    add_extra_ccs: bool,
    date_str: Optional[str] = None,
) -> ComparisonResult:
    """Compare FCMA and Conlloovia for a given scenario.

    Args:
        exp_num: Experiment number.
        napps: Number of apps.
        families: Tuple of instance class families.
        base_cores: Number of cores for the minimum container. Must be 0.120 or 3. The
            actual number of cores will be a value around this number.
        mem_mul: Relation between memory and cores. The actual memory will be around
            actual_cores_app * mem_mul.
        perf: Performance of the minimum container class in the family with the minimum
            price per core. The actual performance will be a value around this and
            proportional to the price.
        frac_gap: Fractional gap for the solver.
        add_extra_ccs: Add extra container classes to the Conlloovia problem.
    """
    fcma_problem = create_fcma_problem(napps, families, base_cores, mem_mul, perf)
    # ProblemPrinter(fcma_problem).print()

    # Save the problem to a pickle file for debugging
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"out/{date_str}"
    json_name = f"fcma_prob_{exp_num}_{napps}_{len(families)}_{base_cores}_{mem_mul}_{perf}_{frac_gap}_{add_extra_ccs}.json"
    os.makedirs(dir_name, exist_ok=True)
    with open(f"{dir_name}/{json_name}", "w") as file:
        problem_serializer = ProblemSerializer(fcma_problem)
        problem_json = problem_serializer.as_dict()
        json.dump(problem_json, file, indent=4)

    conlloovia_problem = Fcma2Conlloovia(fcma_problem).convert(
        add_extra_ccs=add_extra_ccs
    )
    # ProblemPrettyPrinter(conlloovia_problem).print()

    par_names = [
        "exp",
        "napps",
        "n_fam",
        "cores",
        "mem_mul",
        "perf",
        "frac_gap",
        "add_ccs",
    ]
    par_values = [
        exp_num,
        napps,
        len(families),
        base_cores,
        mem_mul,
        perf,
        frac_gap,
        add_extra_ccs,
    ]
    comparator = ComparatorFcmaConlloovia(
        fcma_problem, conlloovia_problem, par_names, par_values, dir_name="out"
    )
    solver_params = SolverParams(
        frac_gap=frac_gap, max_seconds=600, threads=7, seed=150
    )
    res = comparator.compare(solver_params)
    return res


def main() -> None:
    """Main function. Defines the combination of parameters to use in the comparison and
    compares FCMA and Conlloovia for each combination."""
    random.seed(100)  # For reproducibility
    numpy.random.seed(100)  # For reproducibility

    # For saving the intermediary files
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create an output directory for intermediary files
    Path("out").mkdir(parents=True, exist_ok=True)

    fm1 = [aws_eu_west_1.c5_m5_r5_fm]
    fm4 = [
        aws_eu_west_1.c5_m5_r5_fm,
        aws_eu_west_1.c6i_m6i_r6i_fm,
        aws_eu_west_1.c6g_m6g_r6g_fm,
        aws_eu_west_1.c7a_m7a_r7a_fm,
    ]

    l_napps = [1, 2, 5, 15, 30]
    l_fms = [fm1, fm4]
    l_cores = [0.120, 3]
    l_mem_muls = [2, 8]
    l_perfs = [0.02, 0.4]

    frac_gap = 0.02
    add_extra_ccs = False

    comp_results_list = []
    errors = []
    total_combinations = (
        len(l_napps) * len(l_fms) * len(l_cores) * len(l_mem_muls) * len(l_perfs)
    )
    exp_num = 0

    for napps, families, cores, mem_mul, perf in track(
        itertools.product(l_napps, l_fms, l_cores, l_mem_muls, l_perfs),
        total=total_combinations,
    ):
        try:
            res = compare_scenario(
                exp_num,
                napps,
                families,
                cores,
                mem_mul,
                perf,
                frac_gap,
                add_extra_ccs,
                date_str,
            )
            comp_results_list.append(res)

            # Print a table with the comparison and save to CSV the current partial
            # results
            comp_results = ComparisonResults(comp_results_list)
            if comp_results_list:
                print(comp_results.table())
            else:
                print("[red]No results[/red]")

        except Exception as exc:
            _, _, tb = sys.exc_info()
            traceback_info = traceback.extract_tb(tb)
            filename = traceback_info[-1].filename
            line_number = traceback_info[-1].lineno
            function_name = traceback_info[-1].name

            msg = (
                f"Error in exp {exp_num} comparing {napps} apps, {len(families)} "
                f"families, {cores} cores, {mem_mul} mem multiplier, {perf} perf, "
                f"{type(exc).__name__} {exc} in function {function_name} at "
                f"{filename}:{line_number}"
            )
            errors.append(msg)
            print(msg)
            traceback.print_exc()

        exp_num += 1

    # Print a table with the comparison and save CSV
    comp_results = ComparisonResults(comp_results_list)
    if comp_results_list:
        comp_results.save_to_csv(filename="data.csv", out_dir=f".")
        print(comp_results.table())
    else:
        print("[red]No results[/red]")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(err)


if __name__ == "__main__":
    main()
