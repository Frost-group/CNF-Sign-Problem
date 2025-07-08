import matplotlib.pyplot as mp
import numpy as np
import os.path

# Processing parameters
FILES = 1024
RANGE = 4.0
BINS = 128

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

print(f"Found {FILES} data files! Processing data...")

# Choose one particle and the points in range
x_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 1]
y_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 2]

print("Plotting and saving figures...")

# Bin the statistics
counts, ybins, xbins, image = mp.hist2d(
    x_coordinates, y_coordinates, bins=BINS ^ 2)

bar = mp.colorbar(image)

mp.tight_layout()

mp.savefig('src/processing/output/density.png', dpi=300)

mp.clf()

# Only leave the raw image
mp.axis('off')

# Adjust the countours here with the levels argument
figure = mp.contour(counts, extent=[
                    xbins.min(), xbins.max(), ybins.min(), ybins.max()], levels=[20])

# Change the dpi here for a different amount of pixels
mp.savefig('src/processing/output/nodes.png',
           dpi=300, bbox_inches='tight', pad_inches=0)
