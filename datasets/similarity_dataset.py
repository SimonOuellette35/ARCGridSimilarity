from torch.utils.data import Dataset
import numpy as np
import os, json
import random

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


def convertToSimilarity(num_transformations, max_transforms):
    return 1 - num_transformations / max_transforms

# Using this dataset requires cloning the repo: https://github.com/fchollet/ARC
class ARCInspiredSimilarityDataset(Dataset):

    def __init__(self, primitives, grid_dim=5, base_dir="ARC/data/training"):
        self.primitives = primitives
        self.base_dir = base_dir
        self.grid_dim = grid_dim
        self.arc_files = os.listdir(base_dir)
        self.all_grids = []
        self.max_transformations = 8

        self.load_grids()

    def arc_to_numpy(self, fpath):
        with open(fpath) as f:
            content = json.load(f)

        grids = []
        for g in content["train"]:
            grids.append(np.array(g["input"], dtype="int8"))
            grids.append(np.array(g["output"], dtype="int8"))
        for g in content["test"]:
            grids.append(np.array(g["input"], dtype="int8"))
        return grids

    def load_grids(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.base_dir, fname)
            self.all_grids.extend(self.arc_to_numpy(fpath))

    def sampleGridPatch(self):

        min_side = 0
        while min_side < self.grid_dim + 1:
            i = random.randint(0, len(self.all_grids) - 1)
            grid = self.all_grids[i]
            min_side = min(grid.shape[0], grid.shape[1])
        i = random.randint(0, grid.shape[0] - self.grid_dim - 1)
        j = random.randint(0, grid.shape[1] - self.grid_dim - 1)
        start_grid = grid[i:i + self.grid_dim, j:j + self.grid_dim]

        end_grid = np.copy(start_grid)
        num_transformations = random.randint(0, self.max_transformations - 1)
        for _ in range(num_transformations):
            i = np.random.choice(np.arange(len(self.primitives)))
            end_grid = self.primitives[i][0](end_grid)

        return start_grid, end_grid, num_transformations

    def __len__(self):
        return len(self.all_grids)

    def __getitem__(self, idx):
        start_grid, end_grid, y = self.sampleGridPatch()

        flattened_start_grid = np.reshape(start_grid, [1, -1])
        flattened_end_grid = np.reshape(end_grid, [1, -1])

        x = np.concatenate((flattened_start_grid, flattened_end_grid), axis=0)
        y = convertToSimilarity(y, self.max_transformations)
        return [x, np.array([y])]

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

        y = convertToSimilarity(num_transformations, self.MAX_TRANSFORMS)
        return X, np.array([y])

    def __getitem__(self, idx):
        x, y = self.generateGridDistanceSample()

        return [x, y]
