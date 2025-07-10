import numpy as np # The common alias for the numpy module
import math

# Other imports...

def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))

def vector_sigmoid(x: np.ndarray) -> np.ndarray:
  transformed_x = np.zeros(x.shape)

  for i in range(len(x)):
    transformed_x[i] = sigmoid(x[i])

  return transformed_x

def sigmoid_prime(x: float) -> float:
  return sigmoid(x) * (1.0 - sigmoid(x))

def squared_error(expected: float, observed: float) -> float:
  return (expected - observed) ** 2.0

def squared_error_prime(expected: float | np.ndarray, observed: float | np.ndarray) -> float:
  return 2.0 * (observed - expected)

def mean_squared_error(expected: list[float], observed: list[float]) -> float:
  
  assert len(expected) == len(observed)

  sum = 0.0

  for ex, ob in zip(expected, observed):
    sum += (ex - ob) ** 2.0

  return sum / len(expected)

class Network:

  # The shape's first element will be the input layer
  def __init__(self, shape: list[int], learning_rate: float):
    self.weight_matrices = []
    self.weight_gradient = []

    # Biases will be represented by vectors that we can simply add to the output
    # of each weight transformed input
    self.bias_vectors = []
    self.bias_gradient = []

    for i in range(1, len(shape)):
      # The each row represents an output neuron and each column represents an
      # input neuron. Ordering it this way transforms the vector in a way we
      # want
      self.weight_matrices.append(np.random.rand(shape[i], shape[i - 1]))
      
      # We want to start the gradient off with zeros
      self.weight_gradient.append(np.zeros((shape[i], shape[i - 1])))

      self.bias_vectors.append(np.random.rand(shape[i]))
      self.bias_gradient.append(np.zeros(shape[i]))
    
    self.depth = len(shape)
    self.learning_rate = learning_rate

  def feed_forward(self, x: np.ndarray, index: int = 0) -> np.ndarray:

    # Base case; no further transformations to perform
    if index == self.depth - 1:
      return x

    # Weight transformation
    x = np.matmul(self.weight_matrices[index], x)

    # Applying the bias
    x = x + self.bias_vectors[index]

    # Applying activation function
    x = vector_sigmoid(x)

    # Passing it on to the next layer
    return self.feed_forward(x, index + 1)
  
  def apply_gradient(self) -> None:
    for i in range(self.depth - 1):
      self.weight_matrices[i] += -self.learning_rate * self.weight_gradient[i]
      self.bias_vectors[i] += -self.learning_rate * self.bias_gradient[i]

  def zero_gradient(self) -> None:
    for i in range(self.depth - 1):
      self.weight_gradient[i] = np.zeros(self.weight_gradient[i].shape)
      self.bias_gradient[i] = np.zeros(self.bias_gradient[i].shape)
  
  # Other methods...

  # We have to keep track of the z values in order to calculate how the neuron responded to the input
  # and so we can simply activate the z without having to work backwards to find what z was. This is
  # because calculating the derivative of sigmoid requires the unchanged parameter to find what the
  # slope was at that point instead of the activation.
  def calculate_z(self, training_example: np.ndarray) -> list[np.ndarray]:

    # We can't have 2d inputs
    assert training_example.ndim == 1

    z_values = [training_example]

    
    # Our own feed forward that keeps track of the activations

    for transformation, bias in zip(self.weight_matrices, self.bias_vectors):
      training_example = np.matmul(transformation, training_example)
      training_example += bias

      z_values.append(training_example)

      training_example = vector_sigmoid(training_example)

      # No activation function here

    return z_values

  # Only computes the gradient for a single training example; stores result in internal gradient, tuple
  # stores first element as the training input and the second element as the true output
  def calculate_single_gradient(self, training_example: tuple[np.ndarray, np.ndarray], z_values = None,
                                current_influence: float = math.nan, current_neuron: int = 0, depth: int = -1) -> None:

    # Base case, doesn't include input layer
    if depth == self.depth - 1:
      return
    
    # Because function is recursive, state must be stored in the parameters
    if z_values == None:
      z_values = self.calculate_z(training_example[0])
    if current_influence == math.nan:
      current_influence = squared_error_prime(training_example[1], vector_sigmoid(z_values[-1 - depth]))
    if depth == 0:
      assert len(training_example[1]) == len(z_values[-1 - depth])

    # Very inneficient, but covers every path for the gradient

    # We want to perform an initial loop on the first layer of neurons, then allow 
    # recursion to take care of the rest
    if depth == -1:
      for index, z in enumerate(z_values[-1]):
        activation = sigmoid(z)
        influence = squared_error_prime(training_example[1][index], activation)

        self.calculate_single_gradient(training_example, z_values, influence, index, depth + 1)

    else:
      activation_influence = sigmoid_prime(z_values[-1 - depth][current_neuron]) * current_influence

      self.bias_gradient[-1 - depth][current_neuron] += activation_influence

      for index, weight in enumerate(self.weight_matrices[-1 - depth][current_neuron]):
        prev_activation = z_values[-1 - depth][current_neuron]

        # The influence of the weight is determined by the previous activation
        weight_influence = prev_activation * activation_influence

        # The influence of the previous activation is determined by the weight
        prev_activation_influence = weight * activation_influence

        self.weight_gradient[-1 - depth][current_neuron][index] += weight_influence
        self.calculate_single_gradient(training_example, z_values, prev_activation_influence, index, depth + 1)

network = Network([2, 5, 5, 2], 1e-2)
training_example = (np.array([1.0, 1.0]), np.array([0.0, 0.0]))

print("Loss before backprop:", np.sum(squared_error(training_example[1], network.feed_forward(training_example[0]))))

for _ in range(100000):
  network.calculate_single_gradient(training_example)
  network.apply_gradient()
  network.zero_gradient()

print("Loss after backprop:", np.sum(squared_error(training_example[1], network.feed_forward(training_example[0]))))