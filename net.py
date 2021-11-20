from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class StateScoreDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        assert len(self.data) == len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class AgentNet(nn.Module):
    def __init__(self, state_size, n_neurons, activations, lr=0.001):
        nn.Module.__init__(self)
        layers = []
        n_nodes = [state_size] + n_neurons + [1]

        assert len(n_nodes) == len(activations) + 1

        for layer_cnt in range(len(activations)):
            layers.append((f"fc{layer_cnt}", nn.Linear(n_nodes[layer_cnt], n_nodes[layer_cnt + 1])))
            if activations[layer_cnt] == "relu":
                layers.append((f"act{layer_cnt}", nn.ReLU()))

        self.net = nn.Sequential(OrderedDict(layers))
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train(self, data, labels, epochs=3, batch_size=32):
        dataset = StateScoreDataset(data, labels.reshape(-1, 1))
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        for _ in range(epochs):
            for data, label in dataloader:
                self.optimizer.zero_grad()
                loss = self.loss_function(self.net(data), label)
                loss.backward()
                self.optimizer.step()

    def predict(self, data):
        return self.net(data)
