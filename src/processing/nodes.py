import matplotlib.pyplot as mp
import pickle as pl
import numpy as np
import scipy as sp
import os.path

# Processing parameters
FILES = 1024

# Plotting parameters
MOMENTA = 16
RANGE = 4.0
BINS = 1024

print("Reading data files...")

data = False

# Check if there is cache present, otherwise read everything
if os.path.isfile("src/processing/output/density.bin"):
    data = pl.load(open('src/processing/output/density.bin', 'rb'))
else:
    for i in range(FILES):

        print(f"Processing file {i + 1}...")

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

    print(f"Found {FILES} data files!")

print("Processing data...")

# Bin the statistics
coordinate_bins = np.linspace(-RANGE, RANGE, BINS + 1)

binned_signs = sp.stats.binned_statistic_2d(
    data[:, 0], data[:, 1], data[:, 2], bins=[coordinate_bins, coordinate_bins])

binned_deviations = sp.stats.binned_statistic_2d(
    data[:, 0], data[:, 1], data[:, 2], bins=[coordinate_bins, coordinate_bins], statistic='std')

binned_counts = sp.stats.binned_statistic_2d(
    data[:, 0], data[:, 1], None, bins=[coordinate_bins, coordinate_bins], statistic='count')

# Set empty bins to zero and calculate scale bounds
pixel_map = np.nan_to_num(binned_signs.statistic.T)
bound = np.max(np.abs(pixel_map))

print(f"Calculated a maximum bound of {bound}! Plotting and saving figures...")

np.savetxt("src/processing/output/signs.csv", pixel_map, delimiter=", ")

# Only leave the raw image
mp.axis('off')

mp.imshow(pixel_map, cmap='seismic', vmin=-bound, vmax=bound)

mp.savefig('src/processing/output/signs.png', dpi=BINS *
           240 / 887, bbox_inches='tight', pad_inches=0)

mp.clf()

# Plot the relative standard deviation
error_map = binned_deviations.statistic.T / \
    binned_signs.statistic.T / np.sqrt(binned_counts.statistic.T)
error_map = np.nan_to_num(error_map)
error_map = np.abs(error_map)

np.savetxt("src/processing/output/errors.csv", error_map, delimiter=", ")

image = mp.imshow(
    error_map, extent=[-RANGE, RANGE, -RANGE, RANGE], vmin=0.0, vmax=0.2)
mp.colorbar(image)

mp.savefig('src/processing/output/errors.png', dpi=300)

# Calculate momenta

momenta = np.zeros((MOMENTA, 4))

for m in range(MOMENTA):

    momenta_x = data[:, 2] / data[:, 0] ** (m + 1)
    momenta_y = data[:, 2] / data[:, 1] ** (m + 1)

    average_x = np.mean(1 / data[:, 0] ** (m + 1))
    average_y = np.mean(1 / data[:, 1] ** (m + 1))

    momenta[m, 0] = np.mean(momenta_x) / average_x
    momenta[m, 1] = np.std(momenta_x) / average_x / data.shape[0]

    momenta[m, 2] = np.mean(momenta_y) / average_y
    momenta[m, 3] = np.std(momenta_y) / average_y / data.shape[0]

np.savetxt("src/processing/output/momenta.csv", momenta, delimiter=", ")
