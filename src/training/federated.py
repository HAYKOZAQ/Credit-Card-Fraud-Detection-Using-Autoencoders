import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.core.config import Config
from src.core.model import get_model

class FederatedClient:
    def __init__(self, client_id, data, device):
        self.client_id = client_id
        self.device = device
        self.data = data
        self.model = get_model('standard').to(device)
        self.dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(data))
        self.loader = DataLoader(self.dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
    def train(self, global_weights, epochs=1):
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        epoch_loss = 0
        for _ in range(epochs):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        return self.model.state_dict(), len(self.data), epoch_loss / (epochs * len(self.loader))

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def aggregate(self, client_weights, client_sizes):
        global_dict = copy.deepcopy(self.global_model.state_dict())
        
        total_size = sum(client_sizes)
        
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float)
            
        for weights, size in zip(client_weights, client_sizes):
            ratio = size / total_size
            for k in global_dict.keys():
                global_dict[k] += weights[k].clone() * ratio
                
        self.global_model.load_state_dict(global_dict)
        return global_dict

def run_federated_simulation(num_rounds=5):
    print("Initializing Federated Learning Simulation...")
    from src.core.data_loader import get_dataloaders
    train_loader, test_loader, _, _, _, _, _ = get_dataloaders()
    X_train = train_loader.dataset.data.numpy()
    
    num_clients = 3
    clients = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    global_model = get_model('standard').to(Config.DEVICE)
    server = FederatedServer(global_model)
    
    splits = np.array_split(indices, num_clients)
    for i, subset_idx in enumerate(splits):
        client_data = X_train[subset_idx]
        clients.append(FederatedClient(f"Bank_{i+1}", client_data, Config.DEVICE))
        
    history = {"round": [], "client_id": [], "loss": [], "global_val_loss": []}
    
    for round_num in range(num_rounds):
        # Train each client
        client_weights = []
        client_sizes = []
        current_global_weights = server.global_model.state_dict()
        
        for client in clients:
            w, size, loss = client.train(copy.deepcopy(current_global_weights), epochs=1)
            client_weights.append(w)
            client_sizes.append(size)
            history["round"].append(round_num + 1)
            history["client_id"].append(client.client_id)
            history["loss"].append(loss)
            
        server.aggregate(client_weights, client_sizes)
        
        # Evaluate Global Model
        server.global_model.eval()
        val_loss = 0
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = server.global_model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(test_loader)
        
        # Consistent history length
        for _ in range(num_clients):
            history["global_val_loss"].append(val_loss)

    print("Federated Learning Complete.")
    return server.global_model, pd.DataFrame(history)

if __name__ == "__main__":
    run_federated_simulation()
