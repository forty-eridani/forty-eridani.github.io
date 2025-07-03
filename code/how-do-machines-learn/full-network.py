import numpy as np # The common alias for the numpy module
import math

# Other imports...

def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))

def vector_sigmoid(x: np.ndarray) -> np.ndarray:
  transformed_x = np.zeros(x.shape)

  for i in range(len(x)):
    transformed_x[i] = x[i]

  return transformed_x

class Network:

  # The shape's first element will be the input layer
  def __init__(self, shape: list[int]):

    assert len(shape) > 1, "Cannot have network of only one layer"
    
    self.weight_matrices = []

    # Biases will be represented by vectors that we can simply add to the output
    # of each weight transformed input
    self.bias_vectors = []

    for i in range(1, len(shape)):
      # The each row represents an output neuron and each column represents an
      # input neuron. Ordering it this way transforms the vector in a way we
      # want
      self.weight_matrices.append(np.random.randn(shape[i], shape[i - 1]))
      self.bias_vectors.append(np.random.randn(shape[i]))

    self.depth = len(shape)

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
  
network = Network([2, 3, 3, 2])

print(network.feed_forward(np.array([1.0, 2.0])))