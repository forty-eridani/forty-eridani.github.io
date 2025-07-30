import numpy as np
import struct # For interpreting the binary data
import matplotlib.pyplot as plt

def arr_product(arr: list[int]) -> int:
    product = 1

    for element in arr:
        product *= element

    return product

def load_data(filename: str, is_label: bool) -> np.ndarray:

    raw_bin = None

    with open(filename, "rb") as file:
        raw_bin = file.read()

    # Excludes padding bytes
    raw_bin = raw_bin[2:]

    # We will ignore the type since all of our data will use unsigned 8 bit
    # ints. ndim is the amount of dimensions in the data
    ndim = struct.unpack('>BB', raw_bin[:2])[1]

    dim_sizes = []

    for i in range(ndim):
        # A little cursed. struct.unpack gives a goofy array-like thingy, so 
        # I have to index it to get the underlying value
        size = struct.unpack('>I', raw_bin[2 + i * 4:2 + (i + 1) * 4])[0]
        dim_sizes.append(size)

    array = None

    if is_label:
        array = np.array(list(raw_bin[2 + ndim * 4:]), dtype=np.uint8)
    else:
      array = (np.array(list(raw_bin[2 + ndim * 4:]), dtype=np.float32) / 255).reshape(dim_sizes)

    return array