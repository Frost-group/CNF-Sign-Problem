import matplotlib.pyplot as plot
import pickle as pl
import torch as tc
import numpy as np
import os.path

tc.set_default_dtype(tc.float64)
# tc.set_default_device('cuda')
tc.manual_seed(0)


# Simulation parameters
NETWORK_NEURONS = 4
NETWORK_LAYERS = 1
BACKFLOWS = 1
STEP = 1e-3
FILES = 16

# Tolerances
CONVERGENCE_THRESHOLD = 1e-5

# Maps for tensor backflow calculations
BACKFLOW_MAPS = {}

# Network snapshot indexing
SNAPSHOT_INDEX = 0

# Load learning data


def loadData():
    data = np.genfromtxt("src/simulation/output/1.csv", delimiter=',')

    for i in range(FILES - 1):
        if os.path.getsize(
                "src/simulation/output/" + str(i + 2) + ".csv"):
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

# Plot backflow strength and scale network snapshot


def snapshotNetworks(strength_networks, scale_networks):

    global SNAPSHOT_INDEX

    for n in range(BACKFLOWS):
        temperatures = np.linspace(0.8, 1.2, 128)
        signs = np.linspace(0.0, 1.0, 128)

        Y = np.zeros(128)
        X = signs

        for i in range(X.shape[0]):
            Y[i] = strength_networks[n](tc.tensor([signs[i], 1.0])).item()

        plot.plot(X, Y)

        plot.xlabel("Sign Value")
        plot.ylabel("Backflow Strength")

        # plot.ylim(-1.0, 1.0)

        plot.title(
            f"Strength of backflow {n} at learning step {SNAPSHOT_INDEX}")

        plot.savefig(
            f'src/processing/output/strength_{SNAPSHOT_INDEX:09d}.png', dpi=300)

        plot.clf()

        Y = np.zeros(128)
        X = temperatures

        for i in range(X.shape[0]):
            Y[i] = abs(scale_networks[n](
                tc.tensor([temperatures[i]])).item())

        plot.plot(X, Y)

        plot.xlabel("Temperature")
        plot.ylabel("Length Scale")

        # plot.ylim(0.0, 1.0)

        plot.title(
            f"Length scale of backflow {n} at learning step {SNAPSHOT_INDEX}")

        plot.savefig(
            f'src/processing/output/length_{SNAPSHOT_INDEX:09d}.png', dpi=300)

        plot.clf()

        SNAPSHOT_INDEX += 1


if __name__ == "__main__":

    print("Reading input data...")

    # Read distribution data
    data = loadData()

    print(
        f"Calculated average sign of {tc.mean(data[:, 3]).item()} with a standard deviation of {tc.std(data[:, 3]).item()}!")

    # Build backflow coordinate mapping tensor
    buildBackflowMap(data)

    # Check if there is an initial guess present, otherwise setup everything
    if os.path.isfile("src/processing/parameters.bin"):
        strength_networks, scale_networks = pl.load(
            open('src/processing/parameters.bin', 'rb'))
    else:

        strength_networks = []
        scale_networks = []

        # Define the neural networks parameterizing the backflow functions
        for i in range(BACKFLOWS):

            strength_network = [tc.nn.Linear(2, NETWORK_NEURONS, bias=False)]
            scale_network = [tc.nn.Linear(1, NETWORK_NEURONS, bias=False)]

            for layer in range(NETWORK_LAYERS):
                strength_network.extend(
                    [tc.nn.Linear(NETWORK_NEURONS, NETWORK_NEURONS, bias=False), tc.nn.Tanh()])
                scale_network.extend(
                    [tc.nn.Linear(NETWORK_NEURONS, NETWORK_NEURONS, bias=False), tc.nn.Tanh()])

            strength_network.append(tc.nn.Linear(
                NETWORK_NEURONS, 1, bias=False))
            scale_network.append(tc.nn.Linear(NETWORK_NEURONS, 1, bias=False))

            strength_networks.append(tc.nn.Sequential(*strength_network))
            scale_networks.append(tc.nn.Sequential(*scale_network))

    # Use the untransformed cooridnates as the initial guess
    untransformed_coordinates = data[:, 4:7].clone()
    untransformed_coordinates.requires_grad = True

    # Compress the learnable parameters for the optimiser
    parameters = []

    for i in range(BACKFLOWS):
        parameters += list(strength_networks[i].to(
            tc.device(tc.get_default_device())).parameters())
        parameters += list(scale_networks[i].to(
            tc.device(tc.get_default_device())).parameters())

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

        return tc.mean(difference), tc.mean(difference).item()

    previous_average_deviation = np.inf
    previous_log_probability = np.inf
    loop = 0

    # tc.cuda.empty_cache()

    optimizer = tc.optim.AdamW(parameters, lr=STEP)

    print("Learning...")

    # Run constrained optimization with a kill switch
    while True and not os.path.isfile("end"):

        print(f"Running loop {loop}...")

        optimizer.zero_grad()

        # Untransformed coordinate constraint with respect to the untransformed coordinates only
        for parameter in parameters:
            parameter.requires_grad = True

        parameters[-1].requires_grad = True

        cost, average_deviation = data_model_difference(
            untransformed_coordinates)

        relative_change = average_deviation / previous_average_deviation - 1.0
        previous_average_deviation = average_deviation

        cost.backward()

        optimizer.step()

        print(
            f"Enforcing constraint! The average coordinate deviation ({average_deviation}) change is {relative_change}!")

        # tc.cuda.empty_cache()

        # Probability minimisation with respect to the neural networks only, if the constraint condition has been met
        if np.abs(relative_change) < CONVERGENCE_THRESHOLD:

            optimizer.zero_grad()

            for parameter in parameters:
                parameter.requires_grad = True

            parameters[-1].requires_grad = False

            cost, log_probability = probability_minimisation(
                untransformed_coordinates)

            cost.backward()

            optimizer.step()

            # tc.cuda.empty_cache()

            # Snapshot the networks
            snapshotNetworks(strength_networks, scale_networks)

            # Check for convergence
            relative_change = log_probability / previous_log_probability - 1.0
            previous_log_probability = log_probability

            with open("src/processing/output/convergence.csv", "a") as file:
                file.write(f"{loop}, {average_deviation}, {log_probability}\n")

            print(
                f"Coordinate constraint optimised! Relative change of probability ({log_probability}) is {relative_change}!")

            if np.abs(relative_change) < CONVERGENCE_THRESHOLD:
                break

        loop += 1

    print("Complete! Saving networks...")

    # Pickle the networks
    pl.dump([strength_networks, scale_networks], open(
        'src/processing/output/parameters.bin', 'ab'))
