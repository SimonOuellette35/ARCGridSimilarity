import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        device='cuda'
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.device = device

        # LAYERS
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout_p, max_len=51)
        self.src_embedding = nn.Embedding(input_vocab_size, dim_model)
        self.tgt_embedding = nn.Embedding(output_vocab_size, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, output_vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.src_embedding(src) * math.sqrt(self.dim_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

    # This function takes as input a starting grid and a target grid, and returns the similarity.
    def evaluate(self, start_grids, target_grids):

        flattened_start_grids = np.reshape(start_grids, [start_grids.shape[0], -1])
        flattened_target_grids = np.reshape(target_grids, [target_grids.shape[0], -1])

        # TODO: if the colors are from 1 to 10, rather than 0 to 9, should use 0 instead of 10 as separator token
        separators = np.ones([start_grids.shape[0], 1]) * 10

        x_input = np.concatenate((flattened_start_grids, separators, flattened_target_grids), axis=1)

        # SoS token for each sequence in the batch
        sos_tokens = torch.tensor([[0]] * start_grids.shape[0]).to(self.device).long()

        x_input = torch.from_numpy(x_input).to(self.device).long()
        preds = self(x_input, sos_tokens)
        preds = preds.permute(1, 0, 2)

        preds = np.argmax(preds.cpu().data.numpy(), axis=-1)

        # aggregate heuristic values across all k examples.
        return np.mean(preds)