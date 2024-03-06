from model.TransformerModel import Transformer
from datasets.dataset import ARCGymDistanceDataset
import ARC_gym.primitives as primitives
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda'
num_epochs = 100000
lr0 = 0.0001
grid_size = 5

model = Transformer(input_vocab_size=11,
                    output_vocab_size=11,
                    dim_model=64,
                    num_heads=4,
                    num_encoder_layers=5,
                    num_decoder_layers=5,
                    dropout_p=0.).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr0)

prim_functions = primitives.get_total_set()
train_dataset = ARCGymDistanceDataset(prim_functions, grid_size)
train_loader = DataLoader(train_dataset,
                          batch_size=100,
                          shuffle=True)

for epoch in range(num_epochs):
    epoch_train_loss = 0.
    epoch_val_loss = 0.
    num_batches = 0.

    for batch_idx, train_batch in enumerate(train_loader):
        num_batches += 1.

        # x has shape = [batch_size, 2*dim*dim+1], where dim is the width/height of a grid.
        # y has shape = [batch_size]
        x_batch, y_batch = train_batch

        # SoS token for each sequence in the batch
        sos_tokens = torch.tensor([[0]] * y_batch.shape[0], dtype=y_batch.dtype)

        # Prepend SoS tokens to the sequences
        y_sos_batch = torch.cat((sos_tokens, y_batch), dim=1).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y_sos_batch[:, :-1]
        x_batch = x_batch.to(device)
        pred = model(x_batch.long(), y_input.long())

        # Permute pred to have batch size first again
        pred = pred.permute(1, 0, 2)

        pred_flattened = torch.reshape(pred, [pred.shape[0] * pred.shape[1], pred.shape[2]])
        y_flattened = torch.reshape(y_batch, [-1])
        loss = F.cross_entropy(pred_flattened, y_flattened.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.cpu().data.numpy()

    epoch_train_loss /= float(num_batches)

    print("Epoch #%i: %.4f" % (epoch, epoch_train_loss))