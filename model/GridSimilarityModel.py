# Written by François Chollet
# evaluate(grid1, grid2) function added by Simon Ouellette
# 2024-03-07

import keras
import keras_nlp
from keras import ops
import numpy as np

# Expected input format during training:
# x.shape == (1, 2, max_length) - int
#     x[0, 0] is sequence A
#     x[0, 1] is sequence B
# y.shape = (1,) - float - represents transformation distance between A and B

# For inference:
# x.shape == (n, max_length) - int
# returns (n, embedding_dim) - one embedding vector per input sequence

class SimilarityModel(keras.Model):
    def __init__(self, embedding_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.positional_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=15,
            sequence_length=1024,
            embedding_dim=512,
        )
        self.backbone = keras.Sequential([
            keras_nlp.layers.TransformerEncoder(
                intermediate_dim=128,
                num_heads=2,
            ),
            keras_nlp.layers.TransformerEncoder(
                intermediate_dim=128,
                num_heads=2,
            ),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(embedding_dim),
        ])

    def call(self, x):
        if len(x.shape) == 3 and x.shape[0] == 1:
            # Training input format, shape=(1, 2, max_length)
            x = x[0]
        x = self.positional_embedding(x)
        x = self.backbone(x)
        x = ops.normalize(x, axis=-1)
        return x

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs):
        a_embed = y_pred[0]
        b_embed = y_pred[1]
        dot_score = ops.sum(a_embed * b_embed)
        loss = ops.sum(ops.square(dot_score - y[0]))
        return loss

    def get_similarity(self, grid1, grid2):
        '''
        takes as input 2 grids, returns similarity metric
        '''
        batch_sim = 0.
        grid1 = np.reshape(grid1, [grid1.shape[0], grid1.shape[1] * grid1.shape[2]])
        grid2 = np.reshape(grid2, [grid2.shape[0], grid2.shape[1] * grid2.shape[2]])
        for idx in range(grid1.shape[0]):
            # Expected input format: 2, max_length
            a = np.reshape(grid1[0], [1, -1])
            b = np.reshape(grid2[0], [1, -1])
            x = np.concatenate((a,b), axis=0)

            pred = self.call(x)
            similarity = ops.sum(pred[0] * pred[1])

            batch_sim += similarity

        return batch_sim / float(grid1.shape[0])

