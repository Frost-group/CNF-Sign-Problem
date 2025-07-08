import matplotlib.pyplot as mp
import numpy as np

RANGE = 4.0
BINS = 128

print("Reading and processing data file...")

data = np.genfromtxt("src/simulation/output/density.csv", delimiter=',')

x_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 1]
y_coordinates = data[np.logical_and(np.logical_and(data[:, 0] == 0, np.abs(
    data[:, 1]) < RANGE), np.abs(data[:, 2]) < RANGE), 2]

print("Plotting and saving figures...")

counts, ybins, xbins, image = mp.hist2d(
    x_coordinates, y_coordinates, bins=BINS ^ 2)

bar = mp.colorbar(image)

mp.tight_layout()

mp.savefig('src/processing/output/density.png', dpi=300)

mp.clf()

mp.axis('off')

figure = mp.contour(counts, extent=[
                    xbins.min(), xbins.max(), ybins.min(), ybins.max()], levels=[20])

mp.savefig('src/processing/output/nodes.png',
           dpi=96, bbox_inches='tight', pad_inches=0)
