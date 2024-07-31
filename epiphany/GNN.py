import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
from graph_data_loader import GraphDataset
import wandb
import os

wandb.init(project='gnn-hic-prediction')

class symmetrize_bulk(nn.Module):
    def __init__(self):
        super(symmetrize_bulk, self).__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            print("not implemented")
            print(x.shape)
            return None
        else:
            if len(x.shape) == 3:
                a, b, c = x.shape
                x = x.reshape(a, b, 1, c)
                x = x.repeat(1, 1, c, 1)
                x_t = x.permute(0, 1, 3, 2)
                x_sym = torch.concat((x, x_t), axis=1)  # (x+x_t)/2
                return x_sym
            else:
                return None

class EdgeWeightMPNN(MessagePassing):
    def __init__(self, track_channels, track_length, hidden_dim, edge_dim):
        super(EdgeWeightMPNN, self).__init__(aggr='add')
        self.track_length = track_length
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=track_channels, out_channels=16, kernel_size=11, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.linear = nn.Linear(16 * (track_length // 8) + 1, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_predictor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        print("FORWARD")
        print(f"input shape: {data.x.shape}")
        tracks = data.x[:, :-1].reshape(-1, 5, self.track_length)
        print(f"tracks shape: {tracks.shape}")
        pos_enc = data.x[:, -1].unsqueeze(-1)
        print(f"pos_enc shape: {pos_enc.shape}")

        conv_out = torch.relu(self.conv(tracks))
        conv_out = conv_out.view(conv_out.size(0), -1)
        print(f"After conv: {conv_out.shape}")

        node_features = torch.cat([conv_out, pos_enc], dim=1)
        print(f"After concat: {node_features.shape}")

        node_features = torch.relu(self.linear(node_features))
        print(f"After linear: {node_features.shape}")

        out = self.propagate(edge_index=data.edge_index, x=node_features, edge_attr=data.edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        print("MESSAGE")
        edge_attr = edge_attr.unsqueeze(-1)
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        print("UPDATE")
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)

    # def predict_edge_weights(self, x, edge_index):
    #     row, col = edge_index
    #     edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
    #     print(f"edge_embeddings shape: {edge_embeddings.shape}")
    #     edge_weights = self.edge_predictor(edge_embeddings)
    #     print(f"edge_weights shape: {edge_weights.shape}")
    #     return edge_weights.squeeze(-1)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# Parameters for the dataset
window_size = 10000
chroms = ['chr17']
save_dir = '/data/leslie/belova1/Epiphany_dataset'

# Create instances of the custom dataset
train_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)
test_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Model, loss function, optimizer
# track_channels = 5
# track_length = window_size
# hidden_dim = 128
#
# model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=1)

#
# # Move model to GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# torch.manual_seed(0)
# Parameters for the model
track_channels = 5
track_length = 10000  # window_size
hidden_dim = 128
edge_dim = 1

# Initialize the model
model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=edge_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
# Specify the path to the saved model weights
model_path = 'logs/0.1/gnn_hic_prediction_epoch_100.pt'

# Check if the saved model weights exist
if os.path.isfile(model_path):
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    print(f"Model weights loaded from {model_path}")
else:
    print(f"No saved model weights found at {model_path}")

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        edge_weights = batch.edge_attr.squeeze(-1)  # Directly use edge attributes as weights
        loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
    wandb.log({'loss': total_loss / len(train_loader)})

    # Save the model every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'logs/0.1/gnn_hic_prediction_epoch{epoch + 1}.pth')

        # Visualization every 50 epochs
        model.eval()
        all_ground_truth = []
        all_predictions = []
        with torch.no_grad():
            total_test_loss = 0
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                edge_weights = batch.edge_attr.squeeze(-1)
                loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))
                total_test_loss += loss.item()
                all_ground_truth.append(batch.edge_attr.squeeze(-1).cpu().numpy())
                all_predictions.append(edge_weights.cpu().numpy())
            print(f'Test Loss: {total_test_loss / len(test_loader)}')
        all_ground_truth = np.concatenate(all_ground_truth)
        all_predictions = np.concatenate(all_predictions)
        num_nodes = test_dataset[0].x.size(0)
        ground_truth_hic_matrix = test_dataset.contact_maps[chroms[0]]
        predicted_contact_map = np.zeros_like(ground_truth_hic_matrix)
        edge_index = test_dataset[0].edge_index.numpy()
        for idx in range(edge_index.shape[1]):
            i, j = edge_index[:, idx]
            if i < j:
                predicted_contact_map[i, j - i] = all_predictions[idx]
            elif i > j:
                predicted_contact_map[i, i - j] = all_predictions[idx]
            else:
                predicted_contact_map[i, 0] = all_predictions[idx]
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
        im = wandb.Image(plt)
        wandb.log({"Predicted Hi-C Contact Map": im})
        plt.close()
