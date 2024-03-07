from model.TransformerModel import Transformer
from ARC_gym.dataset import ARCGymDataset
from ARC_gym.MetaDGP import MetaDGP
import ARC_gym.primitives as primitives
from ARC_gym.utils.batching import make_gridcoder_batch
import torch
from torch.utils.data import DataLoader
from search.a_star_search import Node, AStarSearch
import time
import copy
import numpy as np

# This script benchmarks the grid-similarity A* search algorithm in comparison to an uninformed but otherwise equivalent
# search (i.e. "brute force search").

#################################### Grid similarity model + hyperparameters ###########################################

device = 'cuda'
grid_size = 5
num_epochs = 1000
k = 25

EMB_DIM = 64
NUM_HEADS = 4
NUM_ENC_LAYERS = 5
NUM_DEC_LAYERS = 5
NUM_MODULES = 12
NUM_EVAL_TASKS = 1000

model = Transformer(input_vocab_size=11,
                    output_vocab_size=11,
                    dim_model=EMB_DIM,
                    num_heads=NUM_HEADS,
                    num_encoder_layers=NUM_ENC_LAYERS,
                    num_decoder_layers=NUM_DEC_LAYERS,
                    dropout_p=0.).to(device)

model.load_state_dict(torch.load('grid_sim_model.pt'))
model = model.double().to(device)

#################################################### Dataset ###########################################################

metadata = {
    'num_nodes': [3, 3],
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

for prim_func, prim_name in primitives.get_total_set()[:NUM_MODULES]:
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
eval_dataset = ARCGymDataset(prim_task, modules, metadata, k, grid_size)

eval_loader = DataLoader(eval_dataset,
                         batch_size=1,
                         collate_fn=lambda x: make_gridcoder_batch(x),
                         shuffle=False)

################################################# Evaluation loop ######################################################

stats = {
    'found1': [],
    'found2': [],
    'num_iterations1': [],
    'num_iterations2': [],
    'elapsed1': [],
    'elapsed2': []
}

for batch_idx, eval_task in enumerate(eval_loader):

    root_node = Node(np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['ys'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     np.reshape(eval_task['xs'][0].cpu().data.numpy(), [k, grid_size, grid_size]),
                     parent_primitive=None,
                     all_primitives=primitives.get_total_set())

    # 1) Run uninformed search
    path = [copy.deepcopy(root_node)]
    time_start = time.time()
    found1, num_iterations1 = AStarSearch.plan(root_node, None)
    time_end = time.time()
    elapsed1 = time_end - time_start

    stats['found1'].append(found1)
    stats['num_iterations1'].append(num_iterations1)
    stats['elapsed1'].append(elapsed1)

    # 2) Run search informed by grid similarity model
    # path = [root_node]
    # time_start = time.time()
    # found2, num_iterations2 = AStarSearch.plan(root_node, model)
    # time_end = time.time()
    # elapsed2 = time_end - time_start
    #
    # stats['found2'].append(found2)
    # stats['num_iterations2'].append(num_iterations2)
    # stats['elapsed2'].append(elapsed2)

    # TODO: 3) confirm success on query example for each solution.

print("=========================================== Evaluation summary ================================================")
print("==> Uninformed search: ")
success_rate = sum(stats['found1']) / float(len(stats['found1']))
print("Success rate at finding solutions: %.2f %%" % (success_rate * 100.))
print("Average elapsed time per task: ", np.mean(stats['elapsed1']))
print("Average number of required node expansions: ", np.mean(stats['num_iterations1']))
# print()
# print("==> Search informed by grid similarity: ")
# success_rate = sum(stats['found2']) / float(len(stats['found2']))
# print("Success rate at finding solutions: %.2f %%" % (success_rate * 100.))
# print("Average elapsed time per task: ", np.mean(stats['elapsed2']))
# print("Average number of required node expansions: ", np.mean(stats['num_iterations2']))