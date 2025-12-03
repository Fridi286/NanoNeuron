# Load the images and scale them down to values between 0 and 1

import numpy as np

def load_images(path):
    with open(path, 'rb') as f:

        # Reading the first 16 bytes of the dataset including information about nums, rows, cols
        f.read(4)
        num = int.from_bytes(f.read(4), 'big')      #num of pics
        rows = int.from_bytes(f.read(4), 'big')     #rows
        cols = int.from_bytes(f.read(4), 'big')     #cols

        data = np.frombuffer(f.read(), dtype=np.uint8)      #remaining datqa contains the pictures (rather values of the numbers)
        data = data.reshape(num, rows * cols)
        return data.astype(np.float32)

# Load Labels

def load_labels(path):
    with open(path, 'rb') as f:
        f.read(4)
        num = int.from_bytes(f.read(4), 'big')

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
