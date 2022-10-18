import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import metrics


def differential_evolution(
    fobj: Callable,
    bounds: np.ndarray,
    args: Optional[Sequence[Any]] = None,
    mutation: float = 0.8,
    recombination: float = 0.7,
    generations: int = 1000,
    strategy: str = "best1bin",
    popsize: int = 20,
    initial_population: Optional[np.ndarray] = None,
    max_dist: Optional[float] = None,
    no_acc: Optional[int] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable] = None,
    cores: int = -1,
    display_progress_bar: bool = False,
):
    """Finds the global maximum of a multivariate function.

    Parameters
    ----------
    fobj : Callable
        The objective function to be maximized. Must be in the form
        f(x, *args), where x is the argument in the form of a 1-D array
        and args is a tuple of any additional fixed parameters needed to
        completely specify the function.

    bounds : Sequence
        Bounds for variables. Sequence with upper and lower bounds, with
        size Nx2, where N is the dimmension of idividuals.

    args : Sequence
        Any additional fixed parameters needed to completely specify the
        objective function.

    mutation : float, optional
        The mutation constant, by default 0.8

    recombination : float, optional
        The recombination constant, by default 0.7

    generations : int, optional
        The maximum number of generations over which the entire population
        is evolved, by default 1000

    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

        - 'best1bin'
        - 'rand1bin'

        The default is 'best1bin'

    popsize : int, optional
        Population size, by default 20

    initial_population : Sequence, optional
        Array specifying the initial population. The array should have
        shape (M, N), where M is the total population size and N is the
        the dimmension of idividuals. Must be normalized. By default is
        None, which means that the population is initialized with Latin
        Hypercube Sampling algorithm.

    max_dist : float, optional
        Stop criteria [1]_: When the average euclidean distance of each
        individual with respect to the best individual is less than a
        threshold th (max_dist), the execution stops.
        By default it is None, which means that this criterion will not
        be used.

    no_acc : int, optional
        Stop criteria [1]_: If not occurred an improvement of the objective
        function in a specified number of generations g (no_acc), the
        execution will stop.
        By default it is None, which means that this criterion will not
        be used.

    seed : int, optional
        Seed used in the process, by default None.

    callback : Callable, optional
        A function to follow the progress of the process, by default None.
        This function will receive a single argument of type
        DifferentialEvolutionSolution.

    cores : int, optional
        Number of cores to use, by default use all cores (-1).

    display_progress_bar : bool, optional
        Display progress bar of process, by default False. Requieres tqdm
        package.

    References
    ----------
    .. [1] Zielinski, K., Weitkemper, P., Laur, R., & Kammeyer, K. D.
            (2006, May). Examination of stopping criteria for differential
            evolution based on a power allocation problem. In Proceedings
            of the 10th International Conference on Optimization of
            Electrical and Electronic Equipment (Vol. 3, pp. 149-156).
    """
    solver = _DifferentialEvolution(
        fobj,
        bounds,
        args=args,
        F=mutation,
        C=recombination,
        generations=generations,
        strategy=strategy,
        popsize=popsize,
        initial_population=initial_population,
        max_dist=max_dist,
        no_acc=no_acc,
        seed=seed,
        callback=callback,
    )  # as solver:

    sol = solver.solve(cores, display_progress_bar)
    return sol


class DifferentialEvolutionSolution:
    def __init__(self, max_gens):
        self.final_population = None
        self.final_fitness = None
        self.total_gens_done = 0
        self.max_gens = max_gens
        self.record_fitness = []
        self.record_mean = []
        self.execution_time = None
        self.stop_by_dist = False
        self.stop_by_iters = False

    def add_generation(self, fitness_values):
        self.record_fitness.append(np.max(fitness_values))
        self.record_mean.append(np.mean(fitness_values))
        self.total_gens_done += 1

    def set_final_values(
        self, population, fitness, total_time, stop_by_dist, stop_by_iters
    ):
        self.final_population = population
        self.final_fitness = fitness

        self.execution_time = total_time
        self.stop_by_dist = stop_by_dist
        self.stop_by_iters = stop_by_iters

    def best_individual(self):
        return self.final_population[np.argmax(self.final_fitness)]

    def best_fitness_value(self):
        return np.max(self.final_fitness)

    def __str__(self):
        out = "Evolutionary solution:\n"
        out += f" * Fitness value:         {self.best_fitness_value():.4f}\n"
        out += f" * Number of generations: {self.total_gens_done} / {self.max_gens}\n"
        out += f" * Execution time:        {self.execution_time:.2f} sec.\n"
        return out


def normalize_population(population, bounds):
    min_b, max_b = np.array(bounds).T
    pop_norm = (population - min_b) / (max_b - min_b)
    pop_norm = np.clip(pop_norm, 0, 1)
    return pop_norm


def init_population_lhs(pop_size, idv_dimm, seed):
    """
    Initializes the population with Latin Hypercube Sampling.
    Latin Hypercube Sampling ensures that each parameter is uniformly
    sampled over its range.
    """
    rng = np.random.RandomState(seed)

    segsize = 1.0 / pop_size
    samples = (
        segsize * rng.uniform(size=(pop_size, idv_dimm))
        + np.linspace(0.0, 1.0, pop_size, endpoint=False)[:, np.newaxis]
    )

    # Create an array for population of candidate solutions.
    population = np.zeros_like(samples)

    for j in range(idv_dimm):
        order = rng.permutation(range(pop_size))
        population[:, j] = samples[order, j]

    return population


class _DifferentialEvolution:
    _STRATEGIES = ["rand1bin", "best1bin"]

    def __init__(
        self,
        fobj: Callable,
        bounds: Sequence,
        args: Sequence = None,
        F: float = 0.8,
        C: float = 0.7,
        generations: int = 1000,
        strategy: str = "best1bin",
        popsize: int = 20,
        initial_population: Sequence = None,
        max_dist: float = None,
        no_acc: int = None,
        seed: int = None,
        callback: Callable = None,
    ):
        """Finds the global maximum of a multivariate function.

        Parameters
        ----------
        fobj : Callable
            The objective function to be maximized. Must be in the form
            f(x, *args), where x is the argument in the form of a 1-D array
            and args is a tuple of any additional fixed parameters needed to
            completely specify the function.

        bounds : Sequence
            Bounds for variables. Sequence with upper and lower bounds, with
            size Nx2, where N is the dimmension of idividuals.

        args : Sequence
            Any additional fixed parameters needed to completely specify the
            objective function.

        F : float, optional
            The mutation constant, by default 0.8

        C : float, optional
            The recombination constant, by default 0.7

        generations : int, optional
            The maximum number of generations over which the entire population
            is evolved, by default 1000

        strategy : str, optional
            The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'rand1bin'

            The default is 'best1bin'

        popsize : int, optional
            Population size, by default 20

        initial_population : Sequence, optional
            Array specifying the initial population. The array should have
            shape (M, N), where M is the total population size and N is the
            the dimmension of idividuals. Must be normalized. By default is
            None, which means that the population is initialized with Latin
            Hypercube Sampling algorithm.

        max_dist : float, optional
            Stop criteria [1]_: When the average euclidean distance of each
            individual with respect to the best individual is less than a
            threshold th (max_dist), the execution stops.
            By default it is None, which means that this criterion will not
            be used.

        no_acc : int, optional
            Stop criteria [1]_: If not occurred an improvement of the objective
            function in a specified number of generations g (no_acc), the
            execution will stop.
            By default it is None, which means that this criterion will not
            be used.

        seed : int, optional
            Seed used in the process, by default None.

        callback : Callable, optional
            A function to follow the progress of the process, by default None.
            This function will receive a single argument of type
            DifferentialEvolutionSolution.

        References
        ----------
        .. [1] Zielinski, K., Weitkemper, P., Laur, R., & Kammeyer, K. D.
               (2006, May). Examination of stopping criteria for differential
               evolution based on a power allocation problem. In Proceedings
               of the 10th International Conference on Optimization of
               Electrical and Electronic Equipment (Vol. 3, pp. 149-156).
        """

        self.fobj = fobj
        self.fobj_args = args
        self.bounds = bounds
        self._dimm = len(bounds)
        self.pop_size = popsize
        self.population = {}
        self.F = F
        self.C = C
        self.iterations = generations
        self.strategy = strategy

        self.random_state = np.random.RandomState(seed)

        self.max_dist = max_dist
        self.no_acc_iters = no_acc
        self.callback = callback

        assert self.max_dist is None or self.max_dist > 0
        assert self.no_acc_iters is None or self.no_acc_iters > 0

        if initial_population is not None:
            assert len(initial_population) == popsize
            self.population = np.array(initial_population, float)
            assert (
                self.population.min() >= 0.0 and self.population.max() <= 1.0
            ), "Las componentes de los individuos deben estar normalizadas en el intervalo [0, 1]"
        else:
            self.population = init_population_lhs(popsize, np.size(bounds, 0), seed)

        self.min_b, self.max_b = np.array(bounds).T
        self.diff = np.fabs(self.min_b - self.max_b)
        self.pop_denorm = self.min_b + self.population * self.diff

        # **
        self.fitness_values = np.array(
            [fobj(ind, *args) for ind in self.pop_denorm], float
        )

        self.best_idx = np.argmax(self.fitness_values)
        self.best_indiviual_denorm = self._denorm(self.population[self.best_idx])

        self.its_since_last_acc = 0
        self.n_cores = mp.cpu_count()

        self.solution = DifferentialEvolutionSolution(self.iterations)

    def _denorm(self, individual):
        return self.min_b + individual * self.diff

    def _evaluate_individual(self, individual_idx: int):

        # 1) Mutate by difference (using F)
        idxs = [idx for idx in range(self.pop_size) if idx != individual_idx]

        mutant = None
        if self.strategy == "rand1bin":
            x1, x2, x3 = self.population[
                self.random_state.choice(idxs, 3, replace=False)
            ]
            mutant = x1 + self.F * (x2 - x3)
        elif self.strategy == "best1bin":
            x1, x2 = self.population[self.random_state.choice(idxs, 2, replace=False)]
            mutant = self.population[self.best_idx] + self.F * (x1 - x2)

        mutant = np.clip(mutant, 0, 1)

        # 2) Recombination (binomial crossover, using C)
        cross_points = self.random_state.rand(self._dimm) < self.C
        if not np.any(cross_points):  # Forzar un mínimo de una caracteristica activada
            cross_points[self.random_state.randint(0, self._dimm)] = True
        trial = np.where(cross_points, mutant, self.population[individual_idx])
        trial_denorm = self.min_b + trial * self.diff

        fitness_trial = self.fobj(trial_denorm, *self.fobj_args)
        return (individual_idx, trial, trial_denorm, fitness_trial)

    def _eval_fobj_multicore(self, individual_idx: int):
        idv = self.population[individual_idx]
        idv_denorm = self.min_b + idv * self.diff

    def _evaluate_initial_fitness(self):
        with mp.Pool(self.n_cores) as p:
            res_gen = p.map_async(
                self._evaluate_individual, np.arange(0, self.pop_size)
            ).get()

    def solve(self, cores=-1, display_progress_bar=False):
        start_time = time.time()
        stop_by_dist = False
        stop_by_iters = False

        self.n_cores = mp.cpu_count() if cores < 1 else min(cores, mp.cpu_count())
        chunksize = int(np.ceil(len(self.population) / float(self.n_cores)))

        progress_bar = tqdm(
            range(1, self.iterations + 1),
            desc=f"Current fitness value: {self.fitness_values[self.best_idx]:.4f}.",
            disable=not display_progress_bar,
        )

        for gen in progress_bar:
            self.its_since_last_acc += 1

            with ThreadPoolExecutor(self.n_cores) as p:
                res_gen = p.map(
                    self._evaluate_individual,
                    np.arange(0, self.pop_size),
                    chunksize=chunksize,
                )

            for res in res_gen:
                idv_idx, trial, trial_denorm, fitness_trial = res

                if fitness_trial > self.fitness_values[idv_idx]:
                    self.fitness_values[idv_idx] = fitness_trial
                    self.population[idv_idx] = trial

                    if fitness_trial > self.fitness_values[self.best_idx]:
                        self.best_idx = idv_idx
                        self.best_indiviual_denorm = trial_denorm
                        self.its_since_last_acc = 0

                        progress_bar.set_description(
                            f"Current fitness value: {fitness_trial:.4f}."
                        )

            self.solution.add_generation(self.fitness_values)

            if self.callback is not None:
                self.callback(self.solution)

            # Stop criterias
            if self.max_dist is not None:
                best_individual = self.population[self.best_idx]
                if np.all(
                    [
                        metrics.euclidean_dist(best_individual, individual)
                        < self.max_dist
                        for individual in self.population
                    ]
                ):
                    stop_by_dist = True
                    break

            if (
                self.no_acc_iters is not None
                and self.its_since_last_acc > self.no_acc_iters
            ):
                stop_by_iters = True
                break

        pop_denorm = self.min_b + self.population * self.diff
        end_time = time.time()
        self.solution.set_final_values(
            pop_denorm,
            self.fitness_values,
            end_time - start_time,
            stop_by_dist,
            stop_by_iters,
        )
        return self.solution
