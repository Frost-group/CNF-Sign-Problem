import torch as tc
import numpy as np


# tc.set_default_device('cuda')

TOLERANCE = 1e-6
STEP = 0.01

# Load learning data


def loadData(filename):
    return tc.tensor(np.genfromtxt(filename, delimiter=','), dtype=tc.cfloat, requires_grad=False)


# Calculate the sign
def calculateSign():
    return 0.0


if __name__ == "__main__":

    print("Reading input data...")

    # Read distribution data
    data = loadData("output.txt")

    print("Learning...")

    parameters = [data]

    # Define the minimization function
    def minimization_function(parameters):

        data = parameters

        return calculateSign(data)

    previous_sign = np.inf
    convergence_data = []
    loop = 0

    # tc.cuda.empty_cache()

    # Run optimization

    optimizer = tc.optim.SGD(parameters, lr=STEP)

    while True:

        sign = minimization_function(parameters)

        optimizer.zero_grad()

        sign.backward()

        optimizer.step()

        # tc.cuda.empty_cache()

        loop += 1

        print(f"Progress is {loop}!")

        # Check for convergence
        sign_change = np.abs(1 - sign.item() / previous_sign)
        print(f"Current sign change is {sign_change}!")

        convergence_data.append(
            [np.real(sign)])

        if sign_change < TOLERANCE:
            break

        previous_sign = sign.item()

    # Save data
    np.savetxt("Convergence Data.csv", np.array(
        convergence_data), delimiter=', ')
