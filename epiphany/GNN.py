import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
from graph_data_loader import GraphDataset

class EdgeWeightMPNN(MessagePassing):
    def __init__(self, track_channels, track_length, hidden_dim, edge_dim):
        super(EdgeWeightMPNN, self).__init__(aggr='add')
        self.track_length = track_length
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=track_channels, out_channels=16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.linear = nn.Linear(800 + 1, hidden_dim)  # Adjusted for reduced dimensions
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Concatenation of aggr_out and x
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_predictor = nn.Linear(hidden_dim * 2, 1)  # Concatenation of row and col features

    def forward(self, data):
        print("FORWARD")
        # Split x into tracks and positional encoding
        tracks = data.x[:, :-1].reshape(-1, 5, self.track_length).cuda()  # Move input to GPU
        pos_enc = data.x[:, -1].unsqueeze(-1).cuda()  # Move input to GPU
        # Apply 1D convolution
        conv_out = torch.relu(self.conv(tracks))  # Shape: [batch_size, hidden_dim, track_length]
        print(f"After bulk_extractor_2d: {conv_out.shape}")
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten to [batch_size, hidden_dim * track_length]
        print(f"After view: {conv_out.shape}")

        # Concatenate positional encoding
        node_features = torch.cat([conv_out, pos_enc], dim=1)  # Shape: [batch_size, hidden_dim * track_length + 1]
        print(f"After concat: {node_features.shape}")

        # Linear layer
        node_features = torch.relu(self.linear(node_features))  # Shape: [batch_size, hidden_dim]
        print(f"After linear: {node_features.shape}")

        # Propagate messages
        out = self.propagate(edge_index=data.edge_index.cuda(), x=node_features, edge_attr=data.edge_attr.cuda())
        return out

    def message(self, x_i, x_j, edge_attr):
        print("MESSAGE")
        edge_attr = edge_attr.unsqueeze(-1)  # Ensure edge_attr has the same number of dimensions
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        print("UPDATE")
        update_input = torch.cat([x, aggr_out], dim=-1)  # Concatenate x and aggr_out
        return self.update_mlp(update_input)

    def predict_edge_weights(self, x, edge_index):
        print("PREDICT_EDGE_WEIGHTS")
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        edge_weights_output = self.edge_predictor(edge_embeddings)
        return edge_weights_output.squeeze(-1)  # Ensure the output is of shape [num_edges]

# Parameters for the dataset
window_size = 10000
chroms = ['chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
save_dir = '/data/leslie/belova1/Epiphany_dataset'

# Create instances of the custom dataset
train_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)
test_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, loss function, optimizer
track_channels = 5  # Number of tracks
track_length = window_size  # Length of each track
hidden_dim = 128*8  # Dimension of hidden layers

model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=1).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Set device for training
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
torch.manual_seed(0)

import wandb
wandb.init(project="gnn-hic-prediction")
wandb.watch(model)  # Watch the EdgeWeightMPNN model we instantiate

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.cuda()  # Move batch to GPU
        optimizer.zero_grad()
        out = model(batch)
        edge_weights = model.predict_edge_weights(out, batch.edge_index)
        loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))  # Ensure the target size matches
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    wandb.log({'loss': total_loss / len(train_loader)})

    # Evaluation on test set and visualize results
    model.eval()
    all_ground_truth = []
    all_predictions = []

    with torch.no_grad():
        total_test_loss = 0
        for batch in test_loader:
            batch = batch.cuda()  # Move batch to GPU
            out = model(batch)
            edge_weights = model.predict_edge_weights(out, batch.edge_index)
            loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))  # Ensure the target size matches
            total_test_loss += loss.item()

            # Collect ground truth and predictions for visualization
            all_ground_truth.append(batch.edge_attr.squeeze(-1).cpu().numpy())
            all_predictions.append(edge_weights.cpu().numpy())

        print(f'Test Loss: {total_test_loss / len(test_loader)}')

        wandb.log({'test_loss': total_test_loss / len(test_loader)})

        # Convert to numpy arrays for visualization
        all_ground_truth = np.concatenate(all_ground_truth)
        all_predictions = np.concatenate(all_predictions)

        # Define num_nodes from test dataset
        num_nodes = test_dataset[0].x.size(0)

        # Extract the ground truth Hi-C matrix from the test dataset
        ground_truth_hic_matrix = test_dataset.contact_maps[chroms[0]]  # Assuming single chromosome in test set

        # Initialize an empty contact map for predictions
        predicted_contact_map = np.zeros_like(ground_truth_hic_matrix)

        # Fill the contact map with predicted edge weights
        edge_index = test_dataset[0].edge_index.numpy()
        predicted_values = all_predictions

        for idx in range(edge_index.shape[1]):
            i, j = edge_index[:, idx]
            if i < j:
                predicted_contact_map[i, j - i] = predicted_values[idx]
            elif i > j:
                predicted_contact_map[i, i - j] = predicted_values[idx]
            else:  # i == j
                predicted_contact_map[i, 0] = predicted_values[idx]

        # Plot the last 400 genomic positions
        n = 400
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.title("Ground Truth Hi-C Contact Map")
        plt.imshow(ground_truth_hic_matrix[:n, :].T, cmap='RdYlBu_r', aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Genomic Position")
        plt.ylabel("Distance to Adjacent Nodes")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Hi-C Contact Map")
        plt.imshow(predicted_contact_map[:n, :].T, cmap='RdYlBu_r', aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Genomic Position")
        plt.ylabel("Distance to Adjacent Nodes")

        plt.tight_layout()
        plt.show()

        ground_truth_image = wandb.Image(plt)
        wandb.log({"Hi-C Contact Map": ground_truth_image})

        plt.close()
