class EnsembleModel:

    def __init__(self, model_list):
        self.model_list = model_list

    def get_similarity(self, grid1, grid2):

        average_sim = 0.0
        for m in self.model_list:
            average_sim += m.get_similarity(grid1, grid2)

        return average_sim / float(len(self.model_list))
