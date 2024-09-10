import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
from graph_data_loader import GraphDataset
import wandb

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
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(1 * (track_length // 8) + 1, hidden_dim)
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
        tracks = data.x[:, :-1].reshape(-1, 5, self.track_length)
        pos_enc = data.x[:, -1].unsqueeze(-1)
        print(f"tracks shape (conv input): {tracks.shape}")
        conv_out = torch.relu(self.conv(tracks))
        print(f"After conv: {conv_out.shape}")
        conv_out = conv_out.view(conv_out.size(0), -1)
        print(f"After conv: {conv_out.shape}")

        node_features = torch.cat([conv_out, pos_enc], dim=1)
        print(f"After concat: {node_features.shape}")

        node_features = torch.relu(self.linear(node_features))
        print(f"After linear: {node_features.shape}")

        out = self.propagate(edge_index=data.edge_index, x=node_features, edge_attr=data.edge_attr)
        return out

    # def message(self, x_i, x_j, edge_attr):
        # print("MESSAGE")
        # print(f"edge_attr before unsqueeze: {edge_attr.shape}")
        # edge_attr = edge_attr.unsqueeze(-1)
        # msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # print(f"edge_attr shape: {edge_attr.shape}")
        # print(f"msg_input shape: {msg_input.shape}")
        # msg_out = self.message_mlp(msg_input)
        # print(f"msg_out shape: {msg_out.shape}")
        # return msg_out
    def message(self, x_i, x_j, edge_attr):
        print("MESSAGE")
        print(f"edge_attr before unsqueeze: {edge_attr.shape}")
        edge_attr = edge_attr.unsqueeze(-1)
        print(f"edge_attr after unsqueeze: {edge_attr.shape}")
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        print(f"msg_input shape: {msg_input.shape}")
        edge_weights = self.edge_predictor(msg_input).squeeze(-1)
        print(f"edge_weights shape: {edge_weights.shape}")
        msg_out = edge_weights * x_j
        print(f"msg_out shape: {msg_out.shape}")
        return msg_out  # Weighted message using edge weights

    def update(self, aggr_out, x):
        print("UPDATE")
        update_input = torch.cat([x, aggr_out], dim=-1)
        print(f"update_input shape: {update_input.shape}")
        update_out = self.update_mlp(update_input)
        print(f"update_out shape: {update_out.shape}")
        return update_out

    def predict_edge_weights(self, x, edge_index):
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        print(f"edge_embeddings shape: {edge_embeddings.shape}")
        edge_weights = self.edge_predictor(edge_embeddings)
        print(f"edge_weights shape: {edge_weights.shape}")
        return edge_weights.squeeze(-1)


# Parameters for the dataset
window_size = 10000
chroms = ['chr17']
save_dir = './Epiphany_dataset'

# Create instances of the custom dataset
train_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)
test_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, loss function, optimizer
track_channels = 5
track_length = window_size
hidden_dim = 128

model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Move model to GPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
torch.manual_seed(0)

# Print which device is being used
if device.type == 'cuda':
    print(f"Running on GPU: {device.index}")
else:
    print("Running on CPU")

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        # Corrected logic: Predict edge weights
        edge_weights = model.predict_edge_weights(out, batch.edge_index)

        # Calculate loss between predicted and ground truth edge weights
        loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
    wandb.log({'loss': total_loss / len(train_loader)})

    # Save model weights
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'logs/gnn_hic_prediction_{epoch + 1}.pt')
        # Visualization
        model.eval()
        all_ground_truth = []
        all_predictions = []
        with torch.no_grad():
            total_test_loss = 0
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                edge_weights = model.predict_edge_weights(out, batch.edge_index)
                loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))
                total_test_loss += loss.item()
                all_ground_truth.append(batch.edge_attr.squeeze(-1).cpu().numpy())
                all_predictions.append(edge_weights.cpu().numpy())
            print(f'Test Loss: {total_test_loss / len(test_loader)}')

        # Convert to numpy arrays for visualization
        all_ground_truth = np.concatenate(all_ground_truth)
        all_predictions = np.concatenate(all_predictions)

        num_nodes = test_dataset[0].x.size(0)
        ground_truth_hic_matrix = test_dataset.contact_maps[chroms[0]]

        predicted_contact_map = np.zeros_like(ground_truth_hic_matrix)

        edge_index = test_dataset[0].edge_index.numpy()
        predicted_values = all_predictions

        for idx in range(edge_index.shape[1]):
            i, j = edge_index[:, idx]
            if i < j:
                predicted_contact_map[i, j - i] = predicted_values[idx]
            elif i > j:
                predicted_contact_map[i, i - j] = predicted_values[idx]
            else:
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

        im = wandb.Image(plt)
        wandb.log({"Predicted Hi-C Contact Map": im})
        plt.close()
# import numpy as np
#
# def extract_off_diagonals_np(matrix, height):
#     n = matrix.shape[0]
#     assert matrix.shape[1] == n, "Input must be a square matrix"
#     assert height <= n, "Height exceeds matrix dimensions"
#
#     result = np.zeros((height, (n - height*2 + 2)*2-1), dtype=matrix.dtype)
#     col_idx = 0
#
#     for i in range(height-1, n - height + 1):
#         diagonal = [matrix[i + j, i - j] for j in range(height)]
#         result[:, col_idx] = diagonal
#         col_idx += 1
#         if i < n - height:
#             diagonal = [matrix[i + j + 1, i - j] for j in range(height)]
#             result[:, col_idx] = diagonal
#             col_idx += 1
#
#     return result
#
# # Example usage
# n = 10
# matrix = np.arange(1, n*n+1).reshape(n, n)
# print("Original Matrix:\n", matrix)
# result_np = extract_off_diagonals_np(matrix, 5)
# print("Extracted Band:\n", result_np)
