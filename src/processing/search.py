import blackbox as bb

from scipy import stats
import numpy as np

import subprocess
import os

# Amount of simulations for a single point
N = 16


def var(parameters):

    # Make sure to not pull the same numbers between different processes
    np.random.seed(os.getpid())

    ids = np.random.randint(2 ** 14, 2**28, N)
    processes = []

    # Run multiple processes for the same backflow parameters, but different seed
    for id in ids:
        processes.append(subprocess.Popen(
            ['./PIMD', str(id), str(parameters[0]), str(parameters[1])]))

    for process_index in range(N):

        # Wait to finish
        processes[process_index].wait()

        # Check if crashed
        if os.path.getsize(f"./output/observables_{str(ids[process_index])}.csv"):

            # Read output
            data_points = np.genfromtxt(
                f"./output/observables_{str(ids[process_index])}.csv", delimiter=',')

            print(
                f"Simulation {str(ids[process_index])} produced {data_points.shape[0]} data points!")

            if data_points.shape[0] != 5000:
                continue

            if 'cumulative_data' not in locals():
                cumulative_data = data_points
            else:
                cumulative_data = np.vstack((cumulative_data, data_points))

        else:
            print(f"Simulation {str(ids[process_index])} crashed!")

    # Cutoff first 10% of data
    thermalisation_cutoff = int(cumulative_data.shape[0] * 0.1)

    adjustment_average = stats.trim_mean(
        cumulative_data[thermalisation_cutoff:, 1], 0.02)
    energy_average = stats.trim_mean(
        cumulative_data[thermalisation_cutoff:, 0], 0.02)

    adjustment_variance = stats.mstats.trimmed_var(
        cumulative_data[thermalisation_cutoff:, 1], 0.02)
    energy_variance = stats.mstats.trimmed_var(
        cumulative_data[thermalisation_cutoff:, 0], 0.02)

    # Calculate variance squared of the energy by removing 2% of outliers
    return energy_average ** 2 / adjustment_average ** 2 * (energy_variance ** 2 / energy_average ** 2 + adjustment_variance ** 2 / adjustment_average**2)


if __name__ == "__main__":
    result = bb.minimize(f=var, domain=[[0.0, -30.0], [
                         0.01, 0.1]], budget=24, batch=6)

    print(
        f"Found best energy variance of value {result['best_f']} at {result['best_x']}!")

    print(
        f"Data was gathered at backflow parameters: \n{result['all_xs']}\nThe corresponding energy variance values were:\n{result['all_fs']}")
