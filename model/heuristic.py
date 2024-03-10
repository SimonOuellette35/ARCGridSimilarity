import numpy as np


class ManualHeuristic:

    def __init__(self):
        pass

    def get_similarity(self, grid1, grid2):
        flat_grid1 = np.reshape(grid1, [-1])
        flat_grid2 = np.reshape(grid2, [-1])

        count_correct = np.count_nonzero(flat_grid1 == flat_grid2)

        return count_correct / float(len(flat_grid1))
