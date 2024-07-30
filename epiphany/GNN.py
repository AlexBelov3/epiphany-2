import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
from graph_data_loader import GraphDataset
import wandb

# Initialize Wandb
wandb.init(project="gnn-hic-prediction")  # Replace 'your_entity' with your Wandb entity name
wandb.config = {
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 2,
    "window_size": 10000
}

class EdgeWeightMPNN(MessagePassing):
    def __init__(self, track_channels, track_length, hidden_dim, edge_dim):
        super(EdgeWeightMPNN, self).__init__(aggr='add')
        self.track_length = track_length
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=track_channels, out_channels=16, kernel_size=11, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32 * (track_length // 8) + 1, hidden_dim)  # Adjust input size for linear layer
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
        tracks = data.x[:, :-1].reshape(-1, 5, self.track_length)
        pos_enc = data.x[:, -1].unsqueeze(-1)
        conv_out = self.conv(tracks)
        conv_out = conv_out.view(conv_out.size(0), -1)
        node_features = torch.cat([conv_out, pos_enc], dim=1)
        node_features = torch.relu(self.linear(node_features))
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

    def predict_edge_weights(self, x, edge_index):
        print("PREDICT_EDGE_WEIGHTS")
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        edge_weights = self.edge_predictor(edge_embeddings)
        return torch.relu(edge_weights.squeeze(-1))  # Ensure the output is non-negative

# Parameters for the dataset
window_size = 10000
chroms = ['chr17']
save_dir = '/data/leslie/belova1/Epiphany_dataset'

# Create instances of the custom dataset
train_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)
test_dataset = GraphDataset(window_size=window_size, chroms=chroms, save_dir=save_dir)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, loss function, optimizer
track_channels = 5
track_length = window_size
hidden_dim = 128

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=1).cuda()
initialize_weights(model)
# model = EdgeWeightMPNN(track_channels=track_channels, track_length=track_length, hidden_dim=hidden_dim, edge_dim=1).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001) #0.01
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        out = model(batch)
        edge_weights = model.predict_edge_weights(out, batch.edge_index)
        loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
    wandb.log({'train_loss': avg_loss})

    # Evaluation on test set and visualize results
    model.eval()
    all_ground_truth = []
    all_predictions = []

    with torch.no_grad():
        total_test_loss = 0
        for batch in test_loader:
            batch = batch.cuda()
            out = model(batch)
            edge_weights = model.predict_edge_weights(out, batch.edge_index)
            loss = loss_fn(edge_weights, batch.edge_attr.squeeze(-1))
            total_test_loss += loss.item()

            # Collect ground truth and predictions for visualization
            all_ground_truth.append(batch.edge_attr.squeeze(-1).cpu().detach().numpy())
            all_predictions.append(edge_weights.cpu().detach().numpy())
        avg_test_loss = total_test_loss / len(test_loader)
        print(f'Test Loss: {avg_test_loss}')
        wandb.log({'test_loss': avg_test_loss})

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
    plt.savefig(f'epoch_{epoch+1}_hic_map.png')
    wandb.log({f"HiC Map": wandb.Image(f'epoch_{epoch+1}_hic_map.png')})
    plt.close()  # Close the figure to free memory
