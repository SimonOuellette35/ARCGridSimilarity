import ARC_gym.primitives as primitives
from torch.utils.data import Dataset
import numpy as np
import re
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

        prim_sequence = []
        grids_so_far = [start_grid]
        end_grid = np.copy(start_grid)
        num_transformations = random.randint(0, self.max_transformations)
        for _ in range(num_transformations):
            valid = False

            # Here we make sure that the transformation doesn't bring us back to a grid configuration we've seen so far
            while not valid:
                i = np.random.choice(np.arange(len(self.primitives)))
                selected_prim = self.primitives[i]
                tmp_grid = selected_prim[0](end_grid)

                # No point in multiple rotations in a single task: redundant.
                valid = True
                if "rotate" in selected_prim[1]:
                    for last_idx in range(len(prim_sequence)):
                        last_prim = prim_sequence[last_idx]
                        if "rotate" in last_prim:
                            valid = False
                            break

                if not valid:
                    continue

                for g in grids_so_far:
                    if np.all(g == tmp_grid):
                        valid = False
                        break

            prim_sequence.append(selected_prim[1])
            end_grid = np.copy(tmp_grid)
            grids_so_far.append(tmp_grid)

        if len(prim_sequence) > 0:
            effective_prim_sequence = self.simplify_sequence(prim_sequence)
        else:
            effective_prim_sequence = prim_sequence

        # re-generate end_grid from start_grid given the effective primitives sequence.
        if len(effective_prim_sequence) > 0:
            end_grid, num_transformations = self.generate_from_sequence(start_grid, effective_prim_sequence)

        return start_grid, end_grid, num_transformations, effective_prim_sequence

    def generate_from_sequence(self, start_grid, prim_sequence):
        end_grid = np.copy(start_grid)
        grids_so_far = [start_grid]

        num_transformations = 0
        for prim_name in prim_sequence:
            prim_func = primitives.fetch_prim_by_name(prim_name)
            tmp_grid = prim_func(end_grid)

            valid = True
            for g in grids_so_far:
                if np.all(g == tmp_grid):
                    valid = False
                    break

            if valid:
                num_transformations += 1
                end_grid = tmp_grid
                grids_so_far.append(tmp_grid)

        return end_grid, num_transformations

    def simplify_sequence(self, prim_sequence):
        shortcuts = primitives.get_shortcuts()
        input_string = '/'.join(prim_sequence)

        # Find all matches for all keys and keep track of them
        matches = []
        for key, value in shortcuts.items():
            for match in re.finditer(re.escape(key), input_string):
                start, end = match.span()
                matches.append((start, end, value))

        # Sort matches by start position; in case of a tie, longer match wins
        matches.sort(key=lambda x: (x[0], -x[1]))

        # Reconstruct the string by replacing as we go, ensuring not to double-replace any part
        result = []
        last_end = 0
        for start, end, replacement in matches:
            if start >= last_end:
                # Add the part of input_string before the current match
                result.append(input_string[last_end:start])

                # Replace the match
                result.append(replacement)
                last_end = end

        # Add any remaining part of input_string after the last match
        result.append(input_string[last_end:])

        # Join all parts together
        effective_prim_sequence = ''.join(result).split("/")
        return effective_prim_sequence

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
