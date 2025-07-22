import matplotlib.pyplot as mp
import pickle as pl
import numpy as np
import scipy as sp
import os.path

# Processing parameters
FILES = 1024

# Plotting parameters
RANGE = 8.0
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

pixel_map = np.nan_to_num(binned_signs.statistic.T)

print("Plotting and saving figures...")

# Only leave the raw image
mp.axis('off')

mp.imshow(pixel_map, interpolation='none', cmap='seismic')

mp.savefig('src/processing/output/nodes.png', dpi=BINS *
           240 / 887, bbox_inches='tight', pad_inches=0)

mp.clf()

image = mp.imshow(binned_deviations.statistic.T,
                  extent=[-RANGE, RANGE, -RANGE, RANGE])
mp.colorbar(image)

mp.savefig('src/processing/output/points.png', dpi=300)
