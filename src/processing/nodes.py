import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import os.path

# Processing parameters
FILES = 1024
BOUND = 1.0
N = 128

print("Reading data files...")

data = False

for i in range(FILES):

    # Check if file is not empty from simulation failure
    if os.path.getsize(
            "src/simulation/output/density_" + str(i + 1) + ".csv"):

        if type(data) == bool:
            data = np.genfromtxt(
                "src/simulation/output/density_" + str(i + 1) + ".csv", delimiter=',')
        else:
            data = np.vstack((data, np.genfromtxt(
                "src/simulation/output/density_" + str(i + 1) + ".csv", delimiter=',')))

    else:
        FILES -= 1

print(f"Found {FILES} data files! Plotting and saving data...")

x_coordinates = data[data[:, 0] == 0, 1]
y_coordinates = data[data[:, 0] == 0, 2]

coordinate_bins = np.linspace(-BOUND, BOUND, N + 1)

binned_density = sp.stats.binned_statistic_2d(
    x_coordinates, y_coordinates, None, statistic='count', bins=[coordinate_bins, coordinate_bins])

statistics = []

for x in range(N):
    for y in range(N):
        statistics.append([-BOUND + BOUND * x / (N - 1), -BOUND +
                          BOUND * y / (N - 1), binned_density.statistic[x, y]])

np.savetxt("statistics.csv", statistics, delimiter=',')

image = mp.imshow(binned_density.statistic.T,
                  origin='lower', extent=[-BOUND, BOUND, -BOUND, BOUND], aspect='auto', cmap='Reds')
bar = mp.colorbar(image)

mp.tight_layout()

mp.savefig('density.png', dpi=300)
