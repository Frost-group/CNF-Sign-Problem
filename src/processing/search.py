import blackbox as bb

from scipy import stats
import numpy as np

import subprocess
import os

# Amount of simulations for a single point
N = 4


def sign(parameters):

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

            if process_index == 0:
                cumulative_data = data_points
            else:
                cumulative_data = np.vstack((cumulative_data, data_points))

        else:
            print(f"Simulation {str(ids[process_index])} crashed!")

    # Cutoff first 10% of data
    thermalisation_cutoff = int(cumulative_data.shape[0] * 0.1)

    # Calculate sign by removing 2% of outliers, flip to maximise
    return -stats.trim_mean(cumulative_data[thermalisation_cutoff:, 1], 0.02) / stats.trim_mean(cumulative_data[thermalisation_cutoff:, 1] / cumulative_data[thermalisation_cutoff:, 2], 0.02)


if __name__ == "__main__":
    result = bb.minimize(f=sign, domain=[[-10.0, -20.0], [
                         0.025, 0.075]], budget=96, batch=24)

    print(
        f"Found best sign of value {-result['best_f']} at {result['best_x']}!")

    print(
        f"Data was gathered at backflow parameters: \n{result['all_xs']}\nThe corresponding flipped sign values were:\n{result['all_fs']}")
