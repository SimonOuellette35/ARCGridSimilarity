import os

os.environ["KERAS_BACKEND"] = "jax"

from ARC_gym.dataset import ARCGymPatchesDataset
from ARC_gym.MetaDGP import MetaDGP
import ARC_gym.utils.visualization as viz
import ARC_gym.primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
from torch.utils.data import DataLoader
from search.a_star_search import Node, AStarSearch
from search.a_star_search_batched import Node as BatchedNode
from search.a_star_search_batched import AStarSearch as BatchedAStarSearch
from model.heuristic import ManualHeuristic
import copy
import numpy as np
import keras

# V2 evaluates the brute force vs A* with pixel-wise heuristic methods on the grid transformation DSL from Michael's
# DSL. This entails varying size grids, and we started with 10x10 grids instead of 5x5.

#################################### Grid similarity model + hyperparameters ###########################################

device = 'cuda'

man_heuristic = ManualHeuristic()

#################################################### Dataset ###########################################################

# TODO


################################################# Evaluation loop ######################################################

stats = {
    'found1': [],
    'found2': [],
    'found3': [],
    'found4': [],
    'success1': [],
    'success2': [],
    'success3': [],
    'success4': [],
    'num_iterations1': [],
    'num_iterations2': [],
    'num_iterations3': [],
    'num_iterations4': [],
    'elapsed1': [],
    'elapsed2': [],
    'elapsed3': [],
    'elapsed4': []
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

batched_planning = BatchedAStarSearch(max_depth=MAX_DEPTH)
planning = AStarSearch(max_depth=MAX_DEPTH)

def search(root_node, tmp_model):
    tmp_root_node = copy.deepcopy(root_node)
    found, program, num_iterations = planning.plan(tmp_root_node, tmp_model)

    success = validate(eval_task, program)

    return found, success, num_iterations

def batched_search(root_node, tmp_model):
    tmp_root_node = copy.deepcopy(root_node)
    found, program, num_iterations = batched_planning.plan(tmp_root_node, tmp_model)

    success = validate(eval_task, program)

    return found, success, num_iterations

for batch_idx, eval_task in enumerate(eval_loader):

    print("Task #%s" % batch_idx)
    print("Trying to solve task %s" % eval_task['task_desc'][0])

    #viz.draw_batch(eval_task, 4)

    support_x = eval_task['xs'][0].cpu().data.numpy()
    support_y = eval_task['ys'][0].cpu().data.numpy()
    root_node = Node(np.reshape(support_x, [k, grid_size, grid_size]),
                     np.reshape(support_y, [k, grid_size, grid_size]),
                     np.reshape(support_x, [k, grid_size, grid_size]),
                     parent_primitive=None,
                     all_primitives=primitives.get_total_set())

    batched_node = BatchedNode(np.reshape(support_x, [k, grid_size, grid_size]),
                     np.reshape(support_y, [k, grid_size, grid_size]),
                     np.reshape(support_x, [k, grid_size, grid_size]),
                     parent_primitive=None,
                     all_primitives=primitives.get_total_set())

    # 1) Run uninformed search
    print("\tUninformed search:")
    found1, success1, num_iterations1 = search(root_node, None)

    if success1:
        stats['num_iterations1'].append(num_iterations1)

    stats['found1'].append(found1)
    stats['success1'].append(success1)

    # 2) Run search informed by grid similarity model
    print("\tGrid similarity-informed search:")
    found2, success2, num_iterations2 = batched_search(batched_node, model)

    if success2:
        stats['num_iterations2'].append(num_iterations2)

    stats['found2'].append(found2)
    stats['success2'].append(success2)

    # 3) Run search informed by manual similarity heuristic
    print("\tManual heuristic-informed search:")
    found3, success3, num_iterations3 = search(root_node, man_heuristic)

    if success3:
        stats['num_iterations3'].append(num_iterations3)

    stats['found3'].append(found3)
    stats['success3'].append(success3)


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
print()
print("==> Search informed by manual heuristic: ")
success_rate = sum(stats['success3']) / float(len(stats['success3']))
print("Success rate at finding solutions: %.2f %%" % (success_rate * 100.))
print("Average number of required node expansions: ", np.mean(stats['num_iterations3']))
