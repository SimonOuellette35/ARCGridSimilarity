from torch.utils.data import Dataset
import numpy as np


# Define a color map
COLOR_MAP = {
    0: 'black',
    1: 'steelblue',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'red',
    7: 'salmon',
    8: 'aquamarine',
    9: 'white'
}

class ARCGymSimilarityDataset(Dataset):

    MAX_TRANSFORMS = 10

    def __init__(self, primitives, grid_size=5):
        self.primitives = primitives
        self.grid_size = grid_size

    def __len__(self):
        return 100

    def generateGrid(self):

        X = np.zeros((self.grid_size, self.grid_size))
        num_px = np.random.choice(np.arange(1, 10))

        pixel_list = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pixel_list.append((y, x))

        pixel_list = np.array(pixel_list)
        pixel_indices = np.random.choice(len(pixel_list), num_px, replace=False)
        x_list = pixel_list[pixel_indices, 0]
        y_list = pixel_list[pixel_indices, 1]
        color_list = np.random.choice(np.arange(10), num_px)

        for i in range(num_px):
            X[x_list[i], y_list[i]] = color_list[i]

        return X

    def generateGridDistanceSample(self):

        # generate X starting grid
        X_start = self.generateGrid()

        # decide on number of transformations to make, between 1 and 10 inclusively
        num_transformations = np.random.choice(np.arange(1, 11))

        # apply sequence of transformations to get X_transformed
        X_end = np.copy(X_start)
        for _ in range(num_transformations):
            transform_idx = np.random.choice(np.arange(len(self.primitives)))
            transform = self.primitives[transform_idx][0]
            X_end = transform(X_end)

        flattened_X_start = np.reshape(X_start, [-1])
        flattened_X_end = np.reshape(X_end, [-1])
        X = np.concatenate((flattened_X_start, [10], flattened_X_end))

        y = self.convertToSimilarity(num_transformations)
        return X, np.array([y])

    def convertToSimilarity(self, num_transformations):
        return 1 - num_transformations / self.MAX_TRANSFORMS

    def __getitem__(self, idx):
        x, y = self.generateGridDistanceSample()

        return [x, y]
