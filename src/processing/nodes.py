import matplotlib.pyplot as mp
import pickle as pl
import numpy as np
import scipy as sp
import os.path

# Processing parameters
FILES = 1024

# Plotting parameters
RANGE = 1.0
BINS = 4096

print("Reading data files...")

data = False

# Check if there is cache present, otherwise read everything
if os.path.isfile("src/processing/output/density.bin"):
    data = pl.load(open('src/processing/output/density.bin', 'rb'))
else:
    for i in range(FILES):

        print(f"Processing file {i}...")

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

    # Cache the data
    pl.dump(data, open(
        f'src/processing/output/density.bin', 'ab'))

print(f"Found {FILES} data files! Processing data...")

# Choose one particle and the points in range
x_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 1]
y_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 2]
sign_values = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 3]

print(f"Average data point density of {sign_values.shape[0] / (BINS ^ 2)}!")

print("Plotting and saving figures...")

# Bin the statistics
coordinate_bins = np.linspace(-RANGE, RANGE, BINS + 1)

binned_density = sp.stats.binned_statistic_2d(
    x_coordinates, y_coordinates, sign_values, bins=[coordinate_bins, coordinate_bins])

image = mp.imshow(binned_density.statistic.T,
                  extent=[-RANGE, RANGE, -RANGE, RANGE])
bar = mp.colorbar(image)

mp.tight_layout()

mp.savefig('src/processing/output/density.png', dpi=300)

mp.clf()

# Only leave the raw image
mp.axis('off')

# Adjust the countours here with the levels argument
figure = mp.contour(binned_density.statistic.T, levels=[0])

# Change the dpi here for a different amount of pixels
mp.savefig('src/processing/output/nodes.png',
           dpi=300, bbox_inches='tight', pad_inches=0)
