import numpy as np
from skimage.metrics import structural_similarity as ssim

class ManualHeuristic:

    def __init__(self):
        pass

    def get_similarity(self, grid1, grid2):
        flat_grid1 = np.reshape(grid1, [-1])
        flat_grid2 = np.reshape(grid2, [-1])

        count_correct = np.count_nonzero(flat_grid1 == flat_grid2)

        return count_correct / float(len(flat_grid1))

class CrossSimilarityHeuristic:

    def __init__(self):
        self.similarity_heuristic = ManualHeuristic()

    def get_similarity(self, grid1, grid2):
        """
        Compute the cross-correlation between two grids and return the maximum correlation score
        and the offset at which it occurs.

        Args:
        - grid1 (np.array): First grid.
        - grid2 (np.array): Second grid, should be of the same size as grid1.

        Returns:
        - max_corr (float): Maximum correlation score.
        - best_offset (tuple): The offset (dy, dx) at which the maximum correlation occurs.
        """
        max_sim = 0.

        # Ensure grid2 is larger for the purpose of cross-correlation
        # TODO: double-check this, looks questionable...
        extended_grid2 = np.pad(grid2, [(grid1.shape[0] - 1, grid1.shape[0] - 1),
                               (grid1.shape[1] - 1, grid1.shape[1] - 1)], mode='constant',
                               constant_values=0)

        # Iterate through all possible translations
        for dy in range(grid1.shape[0] * 2 - 1):
            for dx in range(grid1.shape[1] * 2 - 1):

                # Select the current window from the extended grid2
                window = extended_grid2[dy:dy + grid1.shape[0], dx:dx + grid1.shape[1]]

                # Compute the similarity for this translation
                sim = self.similarity_heuristic.get_similarity(grid1, window)
                if sim > max_sim:
                    max_sim = sim

        return max_sim

class StructualSimilarityIndexHeuristic:

    def __init__(self):
        pass

    # TODO: The output of this function range from -1 (completely different) to 1 (exactly the same).
    def get_similarity(self, grid1, grid2):
        """
        Compute the Structural Similarity Index (SSIM) between two 5x5 grids, with corrected window size.

        Args:
        - grid1 (np.array): First grid.
        - grid2 (np.array): Second grid, both of 5x5 size.

        Returns:
        - ssim_index (float): The SSIM index between the two grids.
        """
        # Correcting the window size for SSIM calculation
        win_size = min(grid1.shape[0], 5)  # Use the actual grid size or 5, whichever is smaller
        data_range = grid2.max() - grid2.min()
        ssim_index = ssim(grid1, grid2, win_size=win_size, data_range=data_range)

        return ssim_index
