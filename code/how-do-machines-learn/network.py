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

def squared_error_prime(expected: float | np.ndarray, observed: float | np.ndarray) -> np.ndarray:
  return 2.0 * (observed - expected)

def sigmoid_prime_activation(a: float) -> float:
  return a * (1 - a)

def mean_squared_error(expected: list[float] | list[np.ndarray], observed: list[float] | list[np.ndarray]) -> float | np.ndarray:
  
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
      self.weight_matrices.append((np.random.rand(shape[i], shape[i - 1]) - 0.5) / 5.0)
      
      # We want to start the gradient off with zeros
      self.weight_gradient.append(np.zeros((shape[i], shape[i - 1])))

      self.bias_vectors.append(np.zeros((shape[i])))
      self.bias_gradient.append(np.zeros(shape[i]))

    self.depth = len(shape)
    self.learning_rate = learning_rate

  def feed_forward(self, x: np.ndarray, index: int = 0) -> np.ndarray:

    x = x.flatten()

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
  def record_activations(self, x: np.ndarray) -> list[np.ndarray]:

    # We can't have 2d inputs
    activations = [x.copy()]

    for i in range(self.depth - 1):
      x = np.matmul(self.weight_matrices[i], x)
      x = x + self.bias_vectors[i]
      x = vector_sigmoid(x)

      activations.append(x.copy())

    return activations


  def calculate_gradient(self, training_data: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for x, y in training_data:

      x = x.flatten()
      a = self.record_activations(x)
      influence = squared_error_prime(y, a[-1])

      for layer in range(self.depth - 1):
        layer = -1 - layer

        self.bias_gradient[layer] += sigmoid_prime_activation(a[layer]) * influence
        tmp_influence = np.zeros(self.weight_gradient[layer].shape[1])

        # influence_matrix = np.array([
        #   [row_influence * sp] * self.weight_gradient[layer].shape[1] for row_influence, sp in zip(influence, sigmoid_prime_activation(a[layer]))
        # ])

        # prev_activation_matrix = np.array([
        #   [prev] * self.weight_gradient[layer].shape[0] for prev in a[layer - 1]
        # ])
        # prev_activation_matrix = np.rot90(prev_activation_matrix)

        # print(prev_activation_matrix.shape, influence_matrix.shape)

        # self.weight_gradient[layer] += influence_matrix * prev_activation_matrix
        # influence = (influence_matrix * self.weight_matrices[layer]).sum(axis=0)

        for neuron_index in range(self.weight_gradient[layer].shape[0]):
          activation = a[layer][neuron_index]

          delta = sigmoid_prime_activation(activation) * influence[neuron_index]
          for prev_neuron_index in range(self.weight_gradient[layer].shape[1]):
            prev_activation = a[layer - 1][prev_neuron_index]

            self.weight_gradient[layer][neuron_index][prev_neuron_index] += prev_activation * delta
            
            tmp_influence[prev_neuron_index] += self.weight_matrices[layer][neuron_index][prev_neuron_index] * delta

        influence = tmp_influence

    for i in range(len(self.weight_gradient)):
      self.weight_gradient[i] /= len(training_data)
      self.bias_gradient[i] /= len(training_data)
