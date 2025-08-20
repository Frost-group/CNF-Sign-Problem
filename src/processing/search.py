import blackbox as bb

from scipy import stats
import numpy as np
import subprocess
import uuid


def sign(parameters):
    id = uuid.uuid4().hex

    output = subprocess.check_output(
        ['src/simulation/PIMD', id, str(parameters[0]), str(parameters[1])])

    print(f"Job {id} produced output: {output}")

    data = np.genfromtxt(
        f"src/simulation/output/observables_{id}.csv", delimiter=',')

    thermalisation_cutoff = int(data.shape[0] * 0.1)

    return -stats.trim_mean(data[thermalisation_cutoff:, 1], 0.02) / stats.trim_mean(data[thermalisation_cutoff:, 1] / data[thermalisation_cutoff:, 2], 0.02)


if __name__ == "__main__":
    result = bb.minimize(f=sign, domain=[[0.0, -30.0], [
                         0.01, 0.1]], budget=384, batch=96)

    print(
        f"Found best sign of value {-result['best_f']} at {result['best_x']}!")

    print(
        f"Data was gathered at backflow parameters {result['all_xs']} with opposite sign values {result['all_fs']}!")
