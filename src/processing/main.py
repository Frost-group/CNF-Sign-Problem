import matplotlib.pyplot as plot
import torch as tc
import numpy as np


# tc.set_default_device('cuda')

# Simulation parameters
NETWORK_DEPTH = 64
TOLERANCE = 1e-3
BACKFLOWS = 2
STEP = 1e-3
FILES = 4

# Load learning data


def loadData():
    data = np.genfromtxt("src/simulation/output/1.csv", delimiter=',')

    for i in range(FILES - 1):
        data = np.vstack((data, np.genfromtxt(
            "src/simulation/output/" + str(i + 2) + ".csv", delimiter=',')))

    return tc.tensor(data, requires_grad=False, dtype=tc.float32)

# Define the hydrodynamical backflow function


def hydrodynamical_backflow_transformation(data, strength_networks, scale_networks, untransformed_coordinates):

    coordinates = untransformed_coordinates.clone()

    for n in range(BACKFLOWS):

        for i in range(data.size()[0]):

            strength = strength_networks[n](data[i, 0:2])
            scale = scale_networks[n](data[i, 0:1])

            for j in range(i % PARTICLES, BEADS * (PARTICLES - 1) + i % PARTICLES, PARTICLES):

                if not i == j:
                    coordinates[i] += strength * (untransformed_coordinates[i] - untransformed_coordinates[j]) / (
                        1.0 + ((untransformed_coordinates[i] - untransformed_coordinates[j]).norm() / scale) ** 3)

    return coordinates

# Define the hydrodynamical backflow log of probability


def hydrodynamical_backflow_probability(data, strength_networks, scale_networks, untransformed_coordinates):

    probability = 0.0

    for n in range(BACKFLOWS):

        for i in range(data.size()[0]):

            strength = strength_networks[n](data[i, 0:2])
            scale = scale_networks[n](data[i, 0:1])

            for j in range(i % PARTICLES, BEADS * (PARTICLES - 1) + i % PARTICLES, PARTICLES):

                if not i == j:
                    probability += strength / \
                        (1.0 + ((untransformed_coordinates[i] -
                         untransformed_coordinates[j]).norm() / scale) ** 3)

                    probability -= 3.0 * strength * ((untransformed_coordinates[i] - untransformed_coordinates[j]).norm() / scale) ** 3 / (
                        1.0 + ((untransformed_coordinates[i] - untransformed_coordinates[j]).norm() / scale) ** 3) ** 2

    return probability


if __name__ == "__main__":

    print("Reading input data...")

    # Read distribution data
    data = loadData()

    global PARTICLES
    global BEADS

    PARTICLES = int(max(data[:, 1]))
    BEADS = int(max(data[:, 2]))

    print("Learning...")

    parameters = []

    # Define the neural networks parameterizing the backflow functions
    strength_networks = []
    scale_networks = []

    for i in range(BACKFLOWS):
        strength_networks.append(tc.nn.Sequential(tc.nn.Linear(
            2, NETWORK_DEPTH), tc.nn.ReLU(), tc.nn.Linear(NETWORK_DEPTH, 1)))
        scale_networks.append(tc.nn.Sequential(tc.nn.Linear(
            1, NETWORK_DEPTH), tc.nn.ReLU(), tc.nn.Linear(NETWORK_DEPTH, 1)))

        parameters += list(strength_networks[i].parameters())
        parameters += list(scale_networks[i].parameters())

    untransformed_coordinates = data[:,
                                     4:7].clone().detach().requires_grad_(True)

    parameters.append(untransformed_coordinates)

    # Define the minimization function
    def minimization_function():

        distribution_fit_cost = tc.sum(tc.abs(data[:, 4:7] - hydrodynamical_backflow_transformation(
            data, strength_networks, scale_networks, untransformed_coordinates)))
        transformation_fit_cost = -tc.sum(hydrodynamical_backflow_probability(
            data, strength_networks, scale_networks, untransformed_coordinates))

        return distribution_fit_cost.item(), transformation_fit_cost.item(), distribution_fit_cost + transformation_fit_cost

    previous_transformation_fit = np.inf
    previous_distribution_fit = np.inf
    convergence_data = []
    loop = 0

    # tc.cuda.empty_cache()

    # Run optimization

    optimizer = tc.optim.Adam(parameters, lr=STEP)

    while True:

        distribution_fit, transformation_fit, cost = minimization_function()

        optimizer.zero_grad()

        cost.backward()

        optimizer.step()

        # tc.cuda.empty_cache()

        loop += 1

        # Check for convergence
        transformation_fit_change = np.abs(
            1 - transformation_fit / previous_transformation_fit)
        distribution_fit_change = np.abs(
            1 - distribution_fit / previous_distribution_fit)

        print(
            f"Transformation fit change after loop {loop} is {transformation_fit_change} and the distribution fit change is {distribution_fit_change}!")

        convergence_data.append(
            [distribution_fit, transformation_fit])

        if transformation_fit_change < TOLERANCE and distribution_fit_change < TOLERANCE:
            break

        previous_transformation_fit = transformation_fit
        previous_distribution_fit = distribution_fit

    # Save data
    np.savetxt("convergence.csv", np.array(convergence_data), delimiter=', ')

    # Plot backflow strengths and scales
    for n in range(BACKFLOWS):
        temperatures = np.linspace(0, 1, 128)
        signs = np.linspace(0, 1, 128)

        X, Y = np.meshgrid(temperatures, signs)
        Z = X + Y

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = strength_networks[n](
                    tc.tensor([temperatures[i], signs[j]], dtype=tc.float32)).item()

        plot.contourf(X, Y, Z, 20, cmap='magma')

        bar = plot.colorbar()
        bar.set_label('Backflow Strength')

        plot.xlabel("Temperature")
        plot.ylabel("Sign Value")

        plot.savefig('strength_' + str(n) + '.png', dpi=300)

        plot.clf()

        Y = tc.zeros(128, dtype=tc.float32)
        X = temperatures

        for i in range(X.shape[0]):
            Y[i] = scale_networks[n](
                tc.tensor([temperatures[i]], dtype=tc.float32)).item()

        plot.plot(X, Y)

        plot.xlabel("Temperature")
        plot.ylabel("Length Scale")

        plot.savefig('length_' + str(n) + '.png', dpi=300)

        plot.clf()
