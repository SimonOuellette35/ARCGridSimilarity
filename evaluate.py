from ARC_gym.dataset import ARCGymPatchesDataset
from ARC_gym.MetaDGP import MetaDGP
import ARC_gym.utils.visualization as viz
import ARC_gym.primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
from torch.utils.data import DataLoader
from search.a_star_search import Node, AStarSearch
from model.heuristic import ManualHeuristic
import time
import copy
import numpy as np
import keras

# ================================================== Results ========================================================
# ==> For model trained on ARCInspiredSimilarityDataset that does not check for redundancies. 22 primitives, k=5
#
# --> Combinatorial complexity analysis (number of node expansions required to solve):
# num_nodes = [4, 4]
# - brute force: 75.08 expansions
# - learned similarity: 62.23 expansions
# - pixel-wise similarity: 31.75 expansions
#
# num_nodes = [5, 5]
# - brute force:  expansions
# - learned similarity:  expansions
# - pixel-wise similarity:  expansions

# --> Temporal complexity analysis (success rate vs given time budget):
# num_nodes = [4, 4], TIMEOUT =
# - brute force: %
# - learned similarity: %
# - pixel-wise similarity: %
#
# num_nodes = [5, 5], TIMEOUT =
# - brute force:  %
# - learned similarity: %
# - pixel-wise similarity: %
#
# num_nodes = [6, 6], TIMEOUT =
# - brute force:  %
# - learned similarity: %
# - pixel-wise similarity: %

# TODO - 1) [TO RUN]: run "success rate" experiments on a time budget for [4, 4], [5, 5], [6, 6], more?
# TODO - 2) [TO RUN]: compare the CrossSimilarityHeuristic on the success rate, is it better than the basic heuristic?
# TODO - 3) [TO RUN]: re-run experiments with DL model trained on ARCInspiredSimilarityDataset that checks for redundancies.
#              -- visualize and validate newly generated data...
# TODO - 4) [TO CODE]: re-run experiments with DL model trained on random grids with checks for redundancies.
#              -- visualize and validate newly generated data...
# TODO - 5) [TO CODE]: evaluate grid similarity model (validation loss) using the get_similarity method. Does the error rate
#  increase as the ground truth similarity gets smaller?
# TODO - 6) [TO TRACE]: What is the impact of scaling h by 10 for informed search?
# TODO - 7) [TO TRACE]: check that given my current implementation, if all children leads to a worse h than the parent, we backtrack
#  (normally A* should do this, make sure my implementation does as well)
# TODO - 8) [TO TEST]: Iterative deepening doesn't seem to work? Goes all the way to MAX_G right away?
# TODO - 9) [TO DESIGN]: a priori model: flat list of probs first, then sequenced/iterative probs.

# ===================================================================================================================

# NOTE: the manual heuristic only works well because of the currently selected grid transformation primitives?

# This script benchmarks the grid-similarity A* search algorithm in comparison to an uninformed but otherwise equivalent
# search (i.e. "brute force search").

#################################### Grid similarity model + hyperparameters ###########################################

device = 'cuda'

model = keras.saving.load_model("best_similarity_model.keras")

man_heuristic = ManualHeuristic()

#################################################### Dataset ###########################################################

#NUM_MODULES = 12
NUM_EVAL_TASKS = 100
MAX_GRAPHS = 1000
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

prim_task = MetaDGP.generateGraphs(metadata, NUM_EVAL_TASKS, MAX_GRAPHS, modules)
eval_dataset = ARCGymPatchesDataset(prim_task, modules, metadata, k, grid_size)

eval_loader = DataLoader(eval_dataset,
                         batch_size=1,
                         collate_fn=lambda x: make_gridcoder_batch(x),
                         shuffle=False)

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

def search(root_node, model):
    tmp_root_node = copy.deepcopy(root_node)
    time_start = time.time()
    found, program, num_iterations = AStarSearch.plan(tmp_root_node, model)
    time_end = time.time()
    elapsed = time_end - time_start

    success = validate(eval_task, program)

    return found, success, num_iterations

for batch_idx, eval_task in enumerate(eval_loader):

    print("Task #%s" % batch_idx)
    print("Trying to solve task %s" % eval_task['task_desc'][0])

    #viz.draw_batch(eval_task, 4)

    root_node = Node(np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['ys'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
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
    found2, success2, num_iterations2 = search(root_node, model)

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
