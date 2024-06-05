# Experiment Overview

The point of the experiment was to explore consistent hyperparameter spaces between nested sampling (NS) implementations
and find the cheapest converging solution to each problem for each NS implementation. The expectation is that a
real-world experimenter would want to choose an NS implementation that converges on their problem with the smallest
computational burden. This is motivated by the fact that faster convergence means more exploration is possible in a
shorter amount of time, which is often crucial to achieving timely scientific results.

# What these results contain

Contains the "reference" posterior for each problem, and the cheapest converging run for each NS implementation, where
convergence is measured with Wasserstein distance. Plots of Wasserstein distance upper bounds and lower bounds for all runs.

## How hyperparameters were chosen

In order for the convergence assessment to be consistent between NS implementations, the hyperparameter spaces were
controlled as much as possible. Since each implementation has different algorithmic
structure, this control was done at a high level using expert knowledge to choose hyperparameter spaces that were mostly
consistent. The following types of spaces were explored:

1. Number of live points, or equivalent concept: not all implementations have the concept of live points, however when
   they did we selected a consistent space of exploration. This space was `{25*D, 50*D, 100*D}`, except for the
   Rastrigin problem which explored `{25*D, 50*D, 100*D, 200*D, 500*D, 1000*D}`.
2. Proposals per acceptance: not all implementations use Markov Chains to sample from the constrained likelihood,
   however when they did, a nominal value of `5*D` accepted proposals per sample was selected.
3. Modes of operation: some implementations have multiple modes of operation, e.g. newer versions of the algorithm. In
   these cases, the top recommended mode(s) were explored, as per private communication with NS authors. However, in
   the case of static vs dynamic nested sampling, static was chosen in order to keep live point spaces consistent.
4. Allowed run time was limited to 1 hour per run, except for the Rastrigin problem which was limited to 2 hours per
   run. This limit was chosen to reflect that in real-world use cases non-expert experimenters may not have the luxury
   of running for long amounts of time, while exploring hyperparameter space.

## How convergence was assessed

The Wasserstein distance for multi-variate distributions is bounded above and below by,

```
max_i W(P_i, Q_i) <= W(P, Q) <= sum_i W(P_i, Q_i)
```

where `P` and `Q` are the reference and estimated posterior distributions, respectively, and `P_i` and `Q_i` are the
marginal distributions of `P` and `Q` respectively.

For each run we computed the Wasserstein distance between the estimated posterior and the reference posterior.
A rule of thumb was used to determine convergence.

Rule of thumb: If the Wasserstein distance upper bound was less than a given threshold, the run was considered to have
converged. The threshold was chosen to be `0.1` for all problems, except for EggBox and Rastrigin which required
slightly higher thresholds.

1. CMB: threshold 0.1
2. Eggbox 10D: threshold 0.3
3. MSSM7: threshold 0.1
4. Rastrigin 10D: threshold 0.2
5. Rosenbrock 10D: threshold 0.1
6. Spikeslab 10D: threshold 0.1

## NS implementations covered

This folder contains a list of directories for each NS implementation and the corresponding cheapest converging run, if
any run converged:

1. `dynesty`
2. `jaxns`
3. `nautilus`
4. `nessai`
5. `pymultinest`
6. `pypolychord`
7. `ultranest`
