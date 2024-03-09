from ARC_gym.dataset import ARCGymPatchesDataset
from ARC_gym.MetaDGP import MetaDGP
import ARC_gym.primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
from torch.utils.data import DataLoader
from search.a_star_search import Node, AStarSearch
import time
import copy
import numpy as np
import keras

# This script benchmarks the grid-similarity A* search algorithm in comparison to an uninformed but otherwise equivalent
# search (i.e. "brute force search").

#################################### Grid similarity model + hyperparameters ###########################################

device = 'cuda'
num_epochs = 1000

model = keras.saving.load_model("best_similarity_model.keras")


#################################################### Dataset ###########################################################

#NUM_MODULES = 12
NUM_EVAL_TASKS = 25
k = 5
grid_size = 5

metadata = {
    'num_nodes': [5, 5],
    'num_pixels': [1, 5],
    'space_dist_x': [0.2, 0.2, 0.2, 0.2, 0.2],
    'space_dist_y': [0.2, 0.2, 0.2, 0.2, 0.2]
}

modules = [
    # input node
    {
        'model': None,
        'name': 'input'
    }
]

for prim_func, prim_name in primitives.get_total_set(): #[:NUM_MODULES]:
    modules.append({
        'model': prim_func,
        'name': prim_name
    })

modules.append(
    # output node
    {
        'model': None,
        'name': 'output'
    }
)

prim_task = MetaDGP.generateGraphs(metadata, NUM_EVAL_TASKS, 100, modules)
eval_dataset = ARCGymPatchesDataset(prim_task, modules, metadata, k, grid_size)

eval_loader = DataLoader(eval_dataset,
                         batch_size=1,
                         collate_fn=lambda x: make_gridcoder_batch(x),
                         shuffle=False)

################################################# Evaluation loop ######################################################

stats = {
    'found1': [],
    'found2': [],
    'success1': [],
    'success2': [],
    'num_iterations1': [],
    'num_iterations2': [],
    'elapsed1': [],
    'elapsed2': []
}

def validate(task, program):

    for k_idx in range(task['xs'][0].shape[0]):
        current_grid = np.reshape(task['xs'][0][k_idx].cpu().data.numpy(), [1, grid_size, grid_size])

        for p in program:
            if p.parent_primitive is None:
                continue

            prim = p.parent_primitive
            prim_func = prim[0]
            current_grid = Node.apply_primitive(prim_func, current_grid)

        a = np.reshape(current_grid, [-1])
        b = np.reshape(task['ys'][0][k_idx].cpu().data.numpy(), [-1])

        if np.any(a != b):
            return False

    # TODO: also validate on the query set
    return True

for batch_idx, eval_task in enumerate(eval_loader):

    print("Task #%s" % batch_idx)
    print("Trying to solve task %s" % eval_task['task_desc'][0])
    root_node = Node(np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['ys'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     parent_primitive=None,
                     all_primitives=primitives.get_total_set())

    # 1) Run uninformed search
    print("\tUninformed search:")
    root_node1 = copy.deepcopy(root_node)
    time_start = time.time()
    found1, program1, num_iterations1 = AStarSearch.plan(root_node1, None)
    time_end = time.time()
    elapsed1 = time_end - time_start

    success1 = validate(eval_task, program1)

    if success1:
        stats['num_iterations1'].append(num_iterations1)

    stats['found1'].append(found1)
    stats['success1'].append(success1)

    # 2) Run search informed by grid similarity model
    print("\tGrid similarity-informed search:")
    root_node2 = copy.deepcopy(root_node)
    time_start = time.time()
    found2, program2, num_iterations2 = AStarSearch.plan(root_node2, model)
    time_end = time.time()
    elapsed2 = time_end - time_start

    success2 = validate(eval_task, program2)

    if success2:
        stats['num_iterations2'].append(num_iterations2)

    stats['found2'].append(found2)
    stats['success2'].append(success2)


print("=========================================== Evaluation summary ================================================")
print("==> Uninformed search: ")
success_rate = sum(stats['success1']) / float(len(stats['success1']))
print("Success rate at finding solutions: %.2f %%" % (success_rate * 100.))
print("Average number of required node expansions: ", np.mean(stats['num_iterations1']))
print()
print("==> Search informed by grid similarity: ")
success_rate = sum(stats['success2']) / float(len(stats['success2']))
print("Success rate at finding solutions: %.2f %%" % (success_rate * 100.))
print("Average number of required node expansions: ", np.mean(stats['num_iterations2']))