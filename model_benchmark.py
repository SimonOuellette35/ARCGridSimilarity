import keras
from datasets.similarity_dataset import ARCInspiredSimilarityDataset
from model.heuristic import ManualHeuristic
import ARC_gym.primitives as primitives
import numpy as np
import time

model = keras.saving.load_model("best_similarity_model.keras")
man_heuristic = ManualHeuristic()

model.quantize('int8')

prim_functions = primitives.get_total_set()
dataset = ARCInspiredSimilarityDataset(prim_functions, 5)

def generate_data_batch(N):

    batch_x = []
    batch_y = []

    for _ in range(N):
        start_grid, end_grid, y, prim_sequence = dataset.sampleGridPatch()

        flattened_start_grid = np.reshape(start_grid, [-1])
        flattened_end_grid = np.reshape(end_grid, [-1])

        batch_x.append(flattened_start_grid)
        batch_y.append(flattened_end_grid)

    return np.array(batch_x), np.array(batch_y)

x_batches = []
y_batches = []

print("Generating data...")
num_examples = 220
for _ in range(1000):
    x_grids, y_grids = generate_data_batch(num_examples)
    x_batches.append(x_grids)
    y_batches.append(y_grids)

print("Benchmarking the learned model...")
model_ts = time.time()
for i in range(1000):
    res = model.get_batched_similarity(x_batches[i], y_batches[i])
model_te = time.time()

print("Benchmarking the pixelwise heuristic...")
heur_ts = time.time()
for i in range(1000):
    for j in range(len(x_batches[i])):
        res = man_heuristic.get_similarity(x_batches[i][j], y_batches[i][j])
heur_te = time.time()

print("Learned model time per batch = ", (model_te - model_ts)/1000.)
print("Pixelwise heuristic time per batch = ", (heur_te - heur_ts)/1000.)
