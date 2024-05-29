import glob
import json
import os
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def measure_wasserstein(samples_1, samples_2):
    """
    Measure the Wasserstein distance between two sets of samples.

    Args:
        samples_1: [N] array of samples
        samples_2: [M] array of samples

    Returns:
        Wasserstein distance
    """
    return wasserstein_distance(samples_1, samples_2)


def compare_results(results_file_1: str, results_file_2: str) -> np.ndarray:
    if not os.path.exists(results_file_1):
        raise FileNotFoundError(f"Results file {results_file_1} not found")
    if not os.path.exists(results_file_2):
        raise FileNotFoundError(f"Results file {results_file_2} not found")

    with open(results_file_1, 'r') as f:
        result_1 = json.load(f)

    with open(results_file_2, 'r') as f:
        result_2 = json.load(f)

    posterior_1 = pd.DataFrame(result_1['posterior']['content'])
    posterior_2 = pd.DataFrame(result_2['posterior']['content'])

    # Iterate through all the parameters, measuring the Wasserstein distance between the dimensions
    wasserstein_distances = []
    dim_names = []
    for dim in posterior_1.columns:
        if not dim.startswith('x'):
            continue
        wasserstein_distances.append(measure_wasserstein(posterior_1[dim], posterior_2[dim]))
        dim_names.append(dim)

    return dim_names, np.asarray(wasserstein_distances)


def get_result_file(result_folder: str) -> str:
    result_files = glob.glob(os.path.join(result_folder, '*_result.json'))
    # print(list(map(lambda s: not s.endswith('experiment_result.json'), result_files)))
    result_files = list(filter(lambda s: not s.endswith('experiment_result.json'), result_files))

    if len(result_files) != 1:
        raise ValueError(
            f"Expected exactly one result file for problem {result_folder}, got {len(result_files)} {result_files}"
        )
    return result_files[0]


def print_latex_tables(metric_results):
    latex_template = r"""
\begin{{table}}[h!]
    \hspace*{{-2.5cm}}
    \tiny
    \centering
    \begin{{tabular}}{{{columns}}}
    \toprule
    Algorithm & {headers} \\\\
    \midrule
{rows}
    \bottomrule
    \end{{tabular}}
    \caption{{Wasserstein distance from reference posterior for {problem}.}}
    \label{{tab:{problem}_wasserstein}}
\end{{table}}
"""
    for problem, data in metric_results.items():
        num_col = None
        for algorithm, results in data.items():
            if results is not None:
                num_col = len(results)
                break
        headers = " & ".join([f"$\\theta_{{{i + 1}}}$" for i in range(num_col)])
        columns = "c" * (num_col + 1)  # 10 parameters + 1 for Algorithm column

        rows = []
        for algorithm, results in data.items():
            if results is None:
                row = [algorithm] + ["--"] * num_col
            else:
                row = [algorithm] + [f"{float(results.get(f'x{i}')):.4f}" for i in range(num_col)]
            rows.append(" & ".join(row) + " \\\\")

        latex_table = latex_template.format(
            problem=problem,
            headers=headers,
            columns=columns,
            rows="\n    \\midrule\n    ".join(rows)
        )
        print(latex_table)




def main(best_posteriors_dir, cheapest_converged_dir, problems_list: List[str], ns_list: List[str]):
    reference_results = dict()
    for problem in problems_list:
        reference_results[problem] = get_result_file(os.path.join(best_posteriors_dir, problem))

    metric_results = dict()  # problem -> ns -> dim_name -> wasserstein_distance
    for ns in ns_list:
        for problem in problems_list:
            print(problem)
            if problem not in metric_results:
                metric_results[problem] = dict()
            if ns not in metric_results[problem]:
                metric_results[problem][ns] = dict()

            try:
                results_file = get_result_file(os.path.join(cheapest_converged_dir, ns, f"{problem}_*"))
            except:
                print(f"Not results found for {problem} with {ns}")
                metric_results[problem][ns] = None
                continue
            dim_names, wasserstein_distances = compare_results(reference_results[problem], results_file)
            print(f"Results for {problem} with {ns} samples")
            for dim_name, dist in zip(dim_names, wasserstein_distances):
                print(f"{dim_name}: {dist:.5f}")
            metric_results[problem][ns] = dict(zip(dim_names, wasserstein_distances))
    print(metric_results)
    print_latex_tables(metric_results=metric_results)


if __name__ == '__main__':
    main(
        best_posteriors_dir='results/reference_posteriors',
        cheapest_converged_dir='./results/cheapest_converged',
        problems_list=['CMB', 'MSSM7', 'eggbox', 'rastrigin', 'rosenbrock', 'spikeslab'],
        ns_list=['dynesty', 'jaxns', 'nautilus', 'nessai', 'pymultinest', 'pypolychord', 'ultranest']
    )
