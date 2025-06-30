import random

class Neuron:
    # If this is an input neuron, there are no input neurons to that
    def __init__(self, weight: float = 0.0, prev_neuron = None, 
                is_input: bool = False):
        
        self.is_input = is_input

        self.weight = weight
        self.prev = prev_neuron

    def activate(self, x: float) -> float:
        if self.is_input:
            # Input neuron only returns the input, otherwise a fairly useless instance
            return x

        # To get the previous neuron's activation, we simple call 
        # the activation function on the previous neuron
        return self.prev.activate(x) * self.weight
        
    def calculate_gradient(self, gradient: list[float], x: float, y: float, 
                                neuron_index: int = 0, current_influence: float = 0.0) -> None:

            # Base case of this recursion; we don't want to perform any logic if this is an input
            if neuron_index == len(gradient):
              return

            # This doesn't have a previous neuron influence as it is the 
            if neuron_index == 0:
              prediction = self.activate(x)
              current_influence = squared_error_prime(y, prediction)

            prev_activation = self.prev.activate(x)

            # Chain rule in action
            weight_influence = prev_activation * current_influence
            prev_activation_influence = self.weight * current_influence

            # Starts at the back of the array
            gradient[-1 - neuron_index] += weight_influence

            # Recursing to the next neuron
            self.calculate_gradient(gradient, x, y, neuron_index + 1, prev_activation_influence)

    # neuron_index will also start at the back like with gradient calculation
    def back_propagate(self, gradient: list[float], learning_rate: float,
                            neuron_index: int = 0) -> None:

        # Once again we have our base case since when this function hits the input,
        # we don't really want much to happen
        if neuron_index == len(gradient):
            return

        self.weight += -learning_rate * gradient[-1 - neuron_index]

        self.prev.back_propagate(gradient, learning_rate, neuron_index + 1)
        
input_neuron = Neuron(is_input=True)

# Remember hidden layers from the earlier section? Well we only have 
# one neuron here, but it acts as a single neuron hidden layer
hidden_neuron = Neuron(0.4, input_neuron)

output_neuron = Neuron(0.7, hidden_neuron)

training_count = 100

training_data = []

for _ in range(100):
    # Let's keep our intended range of values to be [0, 1)
    rand_num = random.random()

    example = (rand_num, rand_num)
    training_data.append(example)

def mean_squared_error(expected: list[float], observed: list[float]) -> float:

    # If the length of the lists don't match, we can't
    # compute the difference between all examples
    assert len(expected) == len(observed)

    n = len(expected)
    sum = 0.0
    for single_expected, single_observed in zip(expected, observed):
        sum += (single_expected - single_observed) ** 2

    mean = sum / n
    return mean

# The explicit mention of prime just indicates this function calculates a derivative
def squared_error_prime(expected: float, observed: float):
    return 2.0 * (observed - expected)

gradient = [0.0, 0.0]

for x, y in training_data:
    output_neuron.calculate_gradient(gradient, x, y)

# Little bit of fun list builder
gradient = [element / len(training_data) for element in gradient]

# Other stuff ...

# Since the array is an array of tuples, we have to split it up 
# for the loss function into two seperate arrays

training_inputs = [e[0] for e in training_data]
expected_outputs = [e[1] for e in training_data]
network_outputs = []


for x in training_inputs:
    network_outputs.append(output_neuron.activate(x))

# Seeing our loss before the network trains
loss = mean_squared_error(expected_outputs, network_outputs)

print(f"Loss before training: {loss}")

epochs = 1000

for _ in range(epochs):

  # We have to clear this array after each epoch or else it will
  # screw up backprop
  gradient = [0.0, 0.0]

  for x, y in training_data:
    output_neuron.calculate_gradient(gradient, x, y)

  gradient = [element / len(training_data) for element in gradient]
  
  for x in training_inputs:
    output_neuron.back_propagate(gradient, 1e-2)

network_outputs = []

for x in training_inputs:
    network_outputs.append(output_neuron.activate(x))

loss = mean_squared_error(expected_outputs, network_outputs)

print(f"Loss after training: {loss}")