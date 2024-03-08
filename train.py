from datasets.similarity_dataset import ARCInspiredSimilarityDataset, ARCGymSimilarityDataset
import ARC_gym.primitives as primitives
import os
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras.callbacks import ModelCheckpoint
from model.GridSimilarityModel import SimilarityModel

device = 'cuda'
LR = 0.0001
grid_size = 5
EMB_DIM = 128

model = SimilarityModel(embedding_dim=EMB_DIM)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=LR))

model_checkpoint_callback = ModelCheckpoint(
    filepath='best_similarity_model.keras',    # Path where to save the model
    save_best_only=True,                    # Only save a model if `val_loss` has improved
    monitor='val_loss',                     # Monitor validation loss
    mode='min',                             # The lower the validation loss, the better
    verbose=1                               # Log a message whenever a model is saved
)

prim_functions = primitives.get_total_set()
dataset = ARCInspiredSimilarityDataset(prim_functions, grid_size)

train_x = []
train_y = []
val_x = []
val_y = []

def convertToSimilarity(num_transformations, max_transforms):
    return 1 - num_transformations / max_transforms

def generate_data_batch(N):

    batch_x = []
    batch_y = []

    for _ in range(N):
        start_grid, end_grid, y = dataset.sampleGridPatch()

        flattened_start_grid = np.reshape(start_grid, [1, -1])
        flattened_end_grid = np.reshape(end_grid, [1, -1])

        x = np.concatenate((flattened_start_grid, flattened_end_grid), axis=0)
        batch_x.append(x)

        y = convertToSimilarity(y, dataset.max_transformations)
        batch_y.append(y)

    return np.array(batch_x), np.array(batch_y)

print("Generating data...")
train_x, train_y = generate_data_batch(100000)
val_x, val_y = generate_data_batch(1000)

print("train_x shape = ", train_x.shape)

print("Baseline validation loss:", 1. - np.average(val_y))

print("Training...")

model.fit(train_x, train_y, epochs=100, batch_size=1, validation_data=(val_x, val_y),
          callbacks=[model_checkpoint_callback])

