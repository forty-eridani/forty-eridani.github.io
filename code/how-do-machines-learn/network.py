import numpy as np # The common alias for the numpy module
import math

# Other imports...

def sigmoid(x: float) -> float:
  return 1 / (1 + np.exp(-x))

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

def sigmoid_prime_from_activation(a: float) -> float:
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

    # List by default stores reference
    activations = [x.copy()]

    for weight, bias in zip(self.weight_matrices, self.bias_vectors):
      x = weight @ x
      x += bias
      x = sigmoid(x)

      activations.append(x.copy())

    return activations

  def calculate_gradient(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> None:
    size = len(batch)
    for x, y in batch:

      x = x.flatten()

      activations = self.record_activations(x)

      delta = squared_error_prime(y, activations[-1]) * sigmoid_prime_from_activation(activations[-1])
      
      # We start at the end of the gradient; stop is exclusive
      for layer in range(self.depth - 2, -1, -1):
      
        # The derivative of z with respect to bias is 1, so no further
        # calculations are needed
        self.bias_gradient[layer] += delta

        # The previous activations are at the current layer since there
        # is one more activation layer than weight/bias transformation 
        # layer
        prev_activations = activations[layer]

        self.weight_gradient[layer] += np.outer(delta, prev_activations)

        delta = np.transpose(self.weight_matrices[layer]) @ delta

    for i in range(self.depth - 1):
      self.bias_gradient[i] /= size
      self.weight_gradient[i] /= size