from ARC_gym.dataset import ARCGymVariableDataset
import ARC_gym.Hodel_primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
from torch.utils.data import DataLoader
import ARC_gym.utils.visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

all_prims = primitives.get_total_set()
metadata = {
    'num_nodes': [3, 3]
}
ds = ARCGymVariableDataset(all_prims, metadata, k=10)

eval_loader = DataLoader(ds,
                         batch_size=1,
                         collate_fn=lambda x: make_gridcoder_batch(x),
                         shuffle=False)

def plot_task(task):
    """ plots a task """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    height = 2
    width = len(task)
    figure_size = (width * 2, height * 2)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    if width == 1:
        axes[0].imshow(task[0]['input'], **args)
        axes[1].imshow(task[0]['output'], **args)
        axes[0].axis('off')
        axes[1].axis('off')
    else:
        for column, example in enumerate(task):
            axes[0, column].imshow(example['input'], **args)
            axes[1, column].imshow(example['output'], **args)
            axes[0, column].axis('off')
            axes[1, column].axis('off')
    figure.set_facecolor('#1E1E1E')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

for batch_idx, eval_task in enumerate(eval_loader):

    print("eval_task['xs'] shape = ", eval_task['xs'].shape)
    task = []
    for k_idx in range(eval_task['xs'].shape[1]):
        example = {
            'input': eval_task['xs'][0][k_idx],
            'output': eval_task['ys'][0][k_idx]
        }
        task.append(example)

    plot_task(task)
