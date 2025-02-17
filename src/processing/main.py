import matplotlib.pyplot as plot
import torch as tc
import numpy as np

tc.set_default_dtype(tc.float64)
# tc.set_default_device('cuda')

# Simulation parameters
NETWORK_NEURONS = 8
NETWORK_LAYERS = 1
BACKFLOWS = 1
STEP = 1e-4
FILES = 128

# Tolerances
CONVERGENCE_THRESHOLD = 1e-3

# Maps for tensor backflow calculations
BACKFLOW_MAPS = {}

# Load learning data


def loadData():
    data = np.genfromtxt("src/simulation/output/1.csv", delimiter=',')

    for i in range(FILES - 1):
        data = np.vstack((data, np.genfromtxt(
            "src/simulation/output/" + str(i + 2) + ".csv", delimiter=',')))

    return tc.tensor(data, requires_grad=False)

# Build maps to calculate the correlated backflow transformation in a tensor way


def buildBackflowMap(data):

    particle_number = int(max(data[:, 1]))
    bead_number = int(max(data[:, 2]))

    indices = np.indices((data.shape[0], particle_number, 3))

    temperature_map = data[indices[0], 0]
    sign_map = data[indices[0], 3]

    # Store temperatures for quick scales neural network calculation
    BACKFLOW_MAPS["scales_network"] = (tc.reshape(
        temperature_map, (data.shape[0] * particle_number * 3, 1)))

    # Store temperatures and signs for quick scales neural network calculation
    BACKFLOW_MAPS["strength_network"] = tc.cat((tc.reshape(
        sign_map, (data.shape[0] * particle_number * 3, 1)), BACKFLOW_MAPS["scales_network"]), 1)

    # Coordinate summation maps
    BACKFLOW_MAPS["origin_vector"] = indices[0, :, :, 0]
    BACKFLOW_MAPS["origin_magnitude"] = indices[0]

    # Apply backflow between particles of the same simulation and bead index
    coordinate_map = (np.floor(
        indices[0] / (particle_number * bead_number)) * (particle_number * bead_number)).astype(int)
    coordinate_map += np.remainder(indices[0], bead_number)
    coordinate_map += indices[1] * particle_number

    BACKFLOW_MAPS["permutation_vector"] = coordinate_map[:, :, 0]
    BACKFLOW_MAPS["permutation_magnitude"] = coordinate_map

    # Don't allow particle self-interactions
    filter_map = (coordinate_map != indices[0]).astype(float)

    BACKFLOW_MAPS["self_filter"] = tc.tensor(
        filter_map, requires_grad=False)

    BACKFLOW_MAPS["zero_probabilities"] = tc.zeros(
        data.shape[0], particle_number)

# Define the hydrodynamical backflow function


def hydrodynamical_backflow_transformation(untransformed_coordinates, strength_networks, scale_networks, particle_number):

    coordinates = untransformed_coordinates[BACKFLOW_MAPS["origin_vector"],
                                            :] / particle_number

    for n in range(BACKFLOWS):

        strengths = tc.reshape(strength_networks[n](
            BACKFLOW_MAPS["strength_network"]), BACKFLOW_MAPS["origin_magnitude"].shape)
        scales = tc.abs(tc.reshape(scale_networks[n](
            BACKFLOW_MAPS["scales_network"]), BACKFLOW_MAPS["origin_magnitude"].shape))

        coordinates += strengths * (untransformed_coordinates[BACKFLOW_MAPS["origin_vector"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_vector"], :]) / (1.0 + (
            (untransformed_coordinates[BACKFLOW_MAPS["origin_magnitude"], :] - untransformed_coordinates[BACKFLOW_MAPS["permutation_magnitude"], :]).norm(dim=3) / scales) ** 3)

    return tc.sum(coordinates, 1)

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

        strength_network = [tc.nn.Linear(2, NETWORK_NEURONS)]
        scale_network = [tc.nn.Linear(1, NETWORK_NEURONS)]

        for layer in range(NETWORK_LAYERS):
            strength_network.extend(
                [tc.nn.Linear(NETWORK_NEURONS, NETWORK_NEURONS), tc.nn.Tanh()])
            scale_network.extend(
                [tc.nn.Linear(NETWORK_NEURONS, NETWORK_NEURONS), tc.nn.Tanh()])

        strength_network.append(tc.nn.Linear(NETWORK_NEURONS, 1))
        scale_network.append(tc.nn.Linear(NETWORK_NEURONS, 1))

        strength_networks.append(tc.nn.Sequential(*strength_network))
        scale_networks.append(tc.nn.Sequential(*scale_network))

        parameters += list(strength_networks[i].parameters())
        parameters += list(scale_networks[i].parameters())

    untransformed_coordinates = data[:, 4:7].clone()
    untransformed_coordinates.requires_grad = True

    parameters.append(untransformed_coordinates)

    # Define the maximum likelihood minimization function
    def probability_minimisation(untransformed_coordinates):

        log_probability = tc.sum(hydrodynamical_backflow_probability(
            strength_networks, scale_networks, untransformed_coordinates))

        return -log_probability, log_probability.item()

    # Define the model and data difference function
    def data_model_difference(untransformed_coordinates):

        difference = tc.abs(data[:, 4:7] - hydrodynamical_backflow_transformation(
            untransformed_coordinates, strength_networks, scale_networks, int(max(data[:, 1]))))

        return tc.mean(difference) / tc.mean(data[:, 4:7]), tc.mean(difference).item() / tc.mean(data[:, 4:7]).item()

    previous_log_probability = np.inf
    convergence_data = []
    loop = 0

    # tc.cuda.empty_cache()

    # Run constrained optimization

    optimizer = tc.optim.AdamW(parameters, lr=STEP)

    while True:

        print("Enforcing constraint...")

        optimizer.zero_grad()

        # Untransformed coordinate constraint with respect to the untransformed coordinates only
        for parameter in parameters:
            parameter.requires_grad = False

        parameters[-1].requires_grad = True

        cost, average_deviation = data_model_difference(
            untransformed_coordinates)

        cost.backward()

        optimizer.step()

        print(
            f"The average coordinate deviation during loop {loop} is {average_deviation}!")

        # tc.cuda.empty_cache()

        # Probability minimisation with respect to the neural networks only, if the constraint condition has been met
        if average_deviation < CONVERGENCE_THRESHOLD:

            print("Coordinate constraint met! Minimising probability...")

            optimizer.zero_grad()

            for parameter in parameters:
                parameter.requires_grad = True

            parameters[-1].requires_grad = False

            cost, log_probability = probability_minimisation(
                untransformed_coordinates)

            cost.backward()

            optimizer.step()

            # tc.cuda.empty_cache()

            print("Checking for total convergence...")

            # Save and print learning data, check for convergence
            convergence_data.append([loop, log_probability])

            relative_change = log_probability / previous_log_probability - 1.0
            previous_log_probability = log_probability

            print(
                f"Relative change of probability ({log_probability}) during loop {loop} is {relative_change}!")

            if np.abs(relative_change) < CONVERGENCE_THRESHOLD:
                break

        loop += 1

    print("Complete! Saving data...")

    # Save data
    np.savetxt("convergence.csv", np.array(convergence_data), delimiter=', ')

    # Plot backflow strengths and scales
    for n in range(BACKFLOWS):
        temperatures = np.linspace(0.8, 1.2, 1024)
        signs = np.linspace(-1.0, 1.0, 1024)

        X, Y = np.meshgrid(temperatures, signs)
        Z = X + Y

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = strength_networks[n](
                    tc.tensor([temperatures[i], signs[j]])).item()

        plot.contourf(X, Y, Z, 20, cmap='magma')

        bar = plot.colorbar()
        bar.set_label('Backflow Strength')

        plot.xlabel("Temperature")
        plot.ylabel("Sign Value")

        plot.savefig('strength_' + str(n) + '.png', dpi=300)

        plot.clf()

        Y = np.zeros(1024)
        X = temperatures

        for i in range(X.shape[0]):
            Y[i] = abs(scale_networks[n](
                tc.tensor([temperatures[i]])).item())

        plot.plot(X, Y)

        plot.xlabel("Temperature")
        plot.ylabel("Length Scale")

        plot.savefig('length_' + str(n) + '.png', dpi=300)

        plot.clf()
