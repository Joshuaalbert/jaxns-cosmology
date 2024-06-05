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


def get_num_likelihood_evaluations(results_folder: str) -> int:
    result_files = glob.glob(os.path.join(results_folder, '*_experiment_result.json'))

    if len(result_files) != 1:
        raise ValueError(
            f"Expected exactly one experiment result file for problem {results_folder}, got {len(result_files)} {result_files}"
        )

    with open(result_files[0], 'r') as fp:
        result = json.load(fp)
    return result['likelihood_evals']


def get_result_file(result_folder: str) -> str:
    result_files = glob.glob(os.path.join(result_folder, '*_result.json'))
    # print(list(map(lambda s: not s.endswith('experiment_result.json'), result_files)))
    result_files = list(filter(lambda s: not s.endswith('experiment_result.json'), result_files))

    if len(result_files) != 1:
        raise ValueError(
            f"Expected exactly one result file for problem {result_folder}, got {len(result_files)} {result_files}"
        )
    return result_files[0]


def main(best_posteriors_dir, hyper_param_search_dir, problems_list: List[str], ns_list: List[str],
         rules_of_thumb: List[float]):
    reference_results = dict()
    for problem in problems_list:
        reference_results[problem] = get_result_file(os.path.join(best_posteriors_dir, problem))

    import pylab as plt

    # make colorwheel of length ns of different colors, so they stand out, colorblind friendly
    colors = plt.cm.tab10.colors

    with open("good_results.txt", "w") as f:
        f.write("#problem likelihood_evals ns run_dir wasserstein_lower wasserstein_upper\n")
    with open("all_results.txt", "w") as f:
        f.write("#problem likelihood_evals ns run_dir wasserstein_lower wasserstein_upper\n")
    for j, problem in enumerate(problems_list):
        rule_of_thumb = rules_of_thumb[j]
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), squeeze=False, sharex=True)
        for i, ns in enumerate(ns_list):
            run_dirs = glob.glob(os.path.join(hyper_param_search_dir, ns, f"{problem}_run_*"))
            for run_dir in run_dirs:
                try:
                    results_file = get_result_file(run_dir)
                    num_likelihood_evals = get_num_likelihood_evaluations(run_dir)
                except:
                    continue
                dim_names, wasserstein_distances = compare_results(reference_results[problem], results_file)
                print(f"Results for {problem} with {ns} samples")
                for dim_name, dist in zip(dim_names, wasserstein_distances):
                    print(f"{dim_name}: {dist:.5f}")
                min_dist = np.max(wasserstein_distances)
                max_dist = np.sum(wasserstein_distances)

                with open("all_results.txt", "a") as f:
                    f.write(f"{problem} {num_likelihood_evals} {ns} {run_dir} {min_dist} {max_dist}\n")

                if max_dist <= rule_of_thumb:
                    with open("good_results.txt", "a") as f:
                        f.write(f"{problem} {num_likelihood_evals} {ns} {run_dir} {min_dist} {max_dist}\n")

                # Plot upper bound
                axs[0][0].scatter(
                    num_likelihood_evals, max_dist, label=ns,
                    c=colors[i]
                )

                # Plot lower bound
                axs[1][0].scatter(
                    num_likelihood_evals, min_dist, label=ns,
                    c=colors[i]
                )
        # Hline in upper bound at rule_of_thumb
        axs[0][0].axhline(rule_of_thumb, color='black', linestyle='--', label='Rule of thumb')
        axs[0][0].set_title(f"{problem} Wasserstein Upper Bound")
        axs[0][0].set_ylabel('Wasserstein distance')
        axs[1][0].set_title(f"{problem} Wasserstein Lower Bound")
        axs[1][0].set_xlabel('Number of likelihood evaluations')
        axs[1][0].set_ylabel('Wasserstein distance')
        # manually set legend, with colors, remove duplicate labels
        handles, labels = axs[0][0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0][0].legend(by_label.values(), by_label.keys(), loc='upper left')
        handles, labels = axs[1][0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1][0].legend(by_label.values(), by_label.keys(), loc='upper left')
        # log scale x axis
        axs[0][0].set_xscale('log')
        axs[1][0].set_xscale('log')
        fig.tight_layout()
        plt.savefig(f'plots/{problem}_wasserstein.png')
        plt.show()


if __name__ == '__main__':
    main(
        best_posteriors_dir='results/reference_posteriors',
        hyper_param_search_dir='./hyperparameter_search',
        problems_list=['CMB', 'MSSM7', 'eggbox', 'rastrigin', 'rosenbrock', 'spikeslab'],
        ns_list=['dynesty', 'jaxns', 'nautilus', 'nessai', 'pymultinest', 'pypolychord', 'ultranest'],
        rules_of_thumb=[0.1, 0.1, 0.3, 0.2, 0.1, 0.1]
    )
