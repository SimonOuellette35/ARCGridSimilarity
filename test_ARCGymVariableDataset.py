from ARC_gym.dataset import ARCGymVariableDataset
import ARC_gym.Hodel_primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
from torch.utils.data import DataLoader
import ARC_gym.utils.visualization as viz

all_prims = primitives.get_total_set()
metadata = {
    'num_nodes': [3, 3]
}
ds = ARCGymVariableDataset(all_prims, metadata, k=10)

eval_loader = DataLoader(ds,
                         batch_size=1,
                         collate_fn=lambda x: make_gridcoder_batch(x),
                         shuffle=False)

for batch_idx, eval_task in enumerate(eval_loader):

    print("eval_task['xs'] shape = ", eval_task['xs'].shape)
    viz.draw_batch(eval_task, 4, grid_shape=[10, 10])