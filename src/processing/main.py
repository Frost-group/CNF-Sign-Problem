import scipy.optimize as optimize
import matplotlib.pyplot as plot
import torch as tc
import numpy as np


# tc.set_default_device('cuda')

# Simulation parameters
NETWORK_DEPTH = 128
BACKFLOWS = 1
STEP = 1e-3
FILES = 2

# Tolerances
LOG_PROBABILITY_TOLERANCE = -np.log(1e6)

# Maps for tensor backflow calculations
BACKFLOW_MAPS = {}

# Load learning data


def loadData():
    data = np.genfromtxt("src/simulation/output/1.csv", delimiter=',')

    for i in range(FILES - 1):
        data = np.vstack((data, np.genfromtxt(
            "src/simulation/output/" + str(i + 2) + ".csv", delimiter=',')))

    return tc.tensor(data, requires_grad=False, dtype=tc.float32)

# Build maps to calculate the correlated backflow transformation in a tensor way


def buildBackflowMap(data):

    global LOG_PROBABILITY_TOLERANCE

    particle_number = int(max(data[:, 1]))
    bead_number = int(max(data[:, 2]))

    # Adjust to number of data points
    LOG_PROBABILITY_TOLERANCE *= particle_number * bead_number

    indices = np.indices((data.shape[0], bead_number, 3))

    temperature_map = data[indices[0], 0]
    sign_map = data[indices[0], 3]

    # Scales neural network map
    BACKFLOW_MAPS["scales_network"] = (tc.reshape(
        temperature_map, (data.shape[0] * bead_number * 3, 1)))

    # Strength neural network map
    BACKFLOW_MAPS["strength_network"] = tc.abs(tc.cat((tc.reshape(
        sign_map, (data.shape[0] * bead_number * 3, 1)), BACKFLOW_MAPS["scales_network"]), 1))

    # Coordinate summation maps
    BACKFLOW_MAPS["origin_vector"] = indices[0, :, :, 0]

    coordinate_map = (np.floor(
        indices[0] / (particle_number * bead_number)) * (particle_number * bead_number)).astype(int)
    coordinate_map += np.remainder(indices[0], bead_number)
    coordinate_map += indices[1] * bead_number

    BACKFLOW_MAPS["permutation_vector"] = coordinate_map[:, :, 0]

    filter_map = (coordinate_map != indices[0]).astype(float)

    BACKFLOW_MAPS["self_filter"] = tc.tensor(
        filter_map, dtype=tc.float32, requires_grad=False)

    BACKFLOW_MAPS["origin_magnitude"] = indices[0]

    BACKFLOW_MAPS["permutation_magnitude"] = coordinate_map

    BACKFLOW_MAPS["zero_probabilities"] = tc.zeros(data.shape[0], bead_number)

# Define the hydrodynamical backflow function


def hydrodynamical_backflow_transformation(untransformed_coordinates, strength_networks, scale_networks, bead_number):

    untransformed_coordinates = tc.tensor(untransformed_coordinates, requires_grad=False, dtype=tc.float32).reshape(
        int(len(untransformed_coordinates) / 3), 3)

    coordinates = untransformed_coordinates[BACKFLOW_MAPS["origin_vector"], :] / bead_number

    for n in range(BACKFLOWS):

        strengths = tc.reshape(strength_networks[n](
            BACKFLOW_MAPS["strength_network"]), BACKFLOW_MAPS["origin_magnitude"].shape)
        scales = tc.abs(tc.reshape(scale_networks[n](
            BACKFLOW_MAPS["scales_network"]), BACKFLOW_MAPS["origin_magnitude"].shape))

        coordinates += strengths * (untransformed_coordinates[BACKFLOW_MAPS["origin_vector"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_vector"], :]) / (1.0 + (
            (untransformed_coordinates[BACKFLOW_MAPS["origin_magnitude"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_magnitude"], :]).norm(dim=3) / scales) ** 3)

    return tc.sum(coordinates, 1).reshape(untransformed_coordinates.shape[0] * 3).detach().numpy()

# Define the hydrodynamical backflow log of probability


def hydrodynamical_backflow_probability(strength_networks, scale_networks, untransformed_coordinates):

    probability = BACKFLOW_MAPS["zero_probabilities"].clone()

    for n in range(BACKFLOWS):

        strengths = tc.reshape(strength_networks[n](
            BACKFLOW_MAPS["strength_network"]), BACKFLOW_MAPS["origin_magnitude"].shape)
        scales = tc.abs(tc.reshape(scale_networks[n](
            BACKFLOW_MAPS["scales_network"]), BACKFLOW_MAPS["origin_magnitude"].shape))

        probability += tc.sum(BACKFLOW_MAPS["self_filter"] * strengths / (1.0 + ((untransformed_coordinates[BACKFLOW_MAPS["origin_magnitude"],
                              :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_magnitude"], :]).norm(dim=3) / scales) ** 3), 2) / 3.0

        probability -= tc.sum(3.0 * strengths * ((untransformed_coordinates[BACKFLOW_MAPS["origin_magnitude"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_magnitude"], :]).norm(dim=3) / scales) ** 3 / (
            1.0 + ((untransformed_coordinates[BACKFLOW_MAPS["origin_magnitude"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_magnitude"], :]).norm(dim=3) / scales) ** 3) ** 2, 2) / 3.0

    return tc.sum(probability, 1)


if __name__ == "__main__":

    print("Reading input data...")

    # Read distribution data
    data = loadData()

    # Build backflow coordinate mapping tensor
    buildBackflowMap(data)

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

    untransformed_coordinates = data[:, 4:7].clone()

    # Define the maximum likelihood minimization function
    def minimisation_function(untransformed_coordinates):

        log_probability = tc.sum(hydrodynamical_backflow_probability(
            strength_networks, scale_networks, untransformed_coordinates))

        return -log_probability

    # Define the model and data difference function
    def data_model_difference(untransformed_coordinates):
        return data[:, 4:7].reshape(data.shape[0] * 3).numpy() - hydrodynamical_backflow_transformation(untransformed_coordinates, strength_networks, scale_networks, int(max(data[:, 2])))

    convergence_data = []
    loop = 0

    # tc.cuda.empty_cache()

    # Run optimization

    optimizer = tc.optim.SGD(parameters, lr=STEP)

    while True:

        print("Solving for untransformed coordinates...")

        untransformed_coordinates = tc.tensor(optimize.fsolve(data_model_difference, untransformed_coordinates.reshape(
            data.shape[0] * 3).numpy()), requires_grad=False, dtype=tc.float32).reshape((data.shape[0], 3))

        print("Minimizing probability...")

        cost = minimisation_function(untransformed_coordinates)

        # Check for convergence
        if cost.item() < LOG_PROBABILITY_TOLERANCE:
            break

        optimizer.zero_grad()

        cost.backward()

        optimizer.step()

        # tc.cuda.empty_cache()

        loop += 1

        print(
            f"The log probability after loop {loop} is {cost.item()}! Targeting {LOG_PROBABILITY_TOLERANCE}...")

        convergence_data.append([loop, cost.item()])

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
            Y[i] = abs(scale_networks[n](
                tc.tensor([temperatures[i]], dtype=tc.float32)).item())

        plot.plot(X, Y)

        plot.xlabel("Temperature")
        plot.ylabel("Length Scale")

        plot.savefig('length_' + str(n) + '.png', dpi=300)

        plot.clf()
