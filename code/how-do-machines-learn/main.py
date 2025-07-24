import numpy as np
import math
import network
import loader
import matplotlib.pyplot as plt

def to_one_hot(n: int, length: int) -> list[float]:
  assert n < length and n >= 0

  arr = [0.0] * length
  arr[n] = 1.0

  return arr

train_imgs = loader.load_data('./data/train-images.idx3-ubyte', False)
# Not in one-hot-encoded form
raw_train_labels = loader.load_data('./data/train-labels.idx1-ubyte', True)
train_labels = [to_one_hot(n, 10) for n in raw_train_labels]

test_imgs = loader.load_data('./data/t10k-images.idx3-ubyte', False)
raw_test_labels = loader.load_data('./data/t10k-labels.idx1-ubyte', True)
test_labels = [to_one_hot(n, 10) for n in raw_test_labels]

print("Data loaded!")

learning_rate = 1.0

model = network.Network([28 * 28, 16, 16, 10], learning_rate)

def train(model: network.Network, batch_size: int, images: np.ndarray, labels: np.ndarray):
  assert len(images) == len(labels)
  size = len(images)

  index = 0

  while index + batch_size < size:
    
    batch = list(zip(images[index:index + batch_size], labels[index:index + batch_size]))

    model.calculate_gradient(batch)
    model.apply_gradient()
    model.zero_gradient()

    preds = []
    expected = []

    for x, y in batch:
      preds.append(model.feed_forward(x))
      expected.append(y)

      # print((x ** 2.0).sum())
      # print(y)
      # print('\n')

    if math.floor(index / batch_size) % 5 == 0:
      print(f"Loss after batch {int(index / batch_size)} / {int(len(images) / batch_size) + 1}: {network.mean_squared_error(expected, preds).sum()}")

    index += batch_size

  batch = list(zip(images[index:], labels[index:]))
  model.calculate_gradient(batch)
  model.apply_gradient()
  model.zero_gradient()

  network_outputs = []
  expected_outputs = list(labels)

  for image in images:
    image = image.flatten()

    network_outputs.append(model.feed_forward(image))

  print("Loss after training:", network.mean_squared_error(expected_outputs, network_outputs).sum())

def test(model: network.Network, images: np.ndarray, labels: np.ndarray):
  assert len(images) == len(labels)

  correct = 0.0

  for x, y in zip(images, labels):
    y = np.argmax(y)
    pred = np.argmax(model.feed_forward(x))

    if y == pred:
      correct += 1.0

  print(f"Total Accuracy: {correct / len(images) * 100.0:.4}%")

epochs = 5

# new_train_data = []
# new_train_labels = []

# model = network.Network([2, 3, 2], 1e-1)

# for i in range(1025):
#   coord = np.random.uniform(-0.7, 0.7, 2)
#   label = np.array([1.0, 0.0]) if (coord ** 2.0).sum() > 0.49 else np.array([0.0, 1.0])

#   new_train_data.append(coord)
#   new_train_labels.append(label)

for epoch in range(epochs):
  print(f"Epoch {epoch}\n---------------------")
  train(model, 64, train_imgs, train_labels)
  print("Testing Data")
  test(model, test_imgs, test_labels)