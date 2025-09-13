# from lux.lux import LUX
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
from torch import nn, optim
import torch

class SimpleNN(nn.Module):
    def __init__(self,input_size, num_classes = 1):
        super(SimpleNN, self).__init__()

        self.decision_threshold = 0.5
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)
        # self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x

    def fit(self, X, y):
        device = next(self.parameters()).device
        # converting data to tensors
        X_train_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(
            y.values if isinstance(y, pd.Series) else y,
            dtype=torch.float32, device=device
        )
        if y_train_tensor.ndim == 1:
            y_train_tensor = y_train_tensor.view(-1, 1)

        # loss and optimizer
        pos = (y_train_tensor == 1).sum()
        neg = (y_train_tensor == 0).sum()
        pos_weight = torch.tensor([1.0], device=device) if pos.item() == 0 else (neg / pos).to(device).view(1)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, verbose=True
        )

        best_loss = float('inf')

        print(f"First few samples: {X[:5]}")
        print(f"Target distributions: {np.bincount(y.astype(np.int64))}")
        epochs = 1000

        # training loop
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train_tensor)

            if epoch == 0:
                print(f"Initial outputs: {outputs[:5]}")
                print(f"Initial targets: {y_train_tensor[:5]}")

            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            if epoch % 10 == 0:
                with torch.no_grad():
                    grad_norm = sum(
                        p.grad.norm().item()
                        for p in self.parameters()
                        if p.grad is not None
                    )
                    print(f"Gradient norm: {grad_norm}")

                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    accuracy = (predictions == y_train_tensor).float().mean().item()
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], "
                        f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                    )



    def predict_proba(self, X):
        if hasattr(X, "toarray"):  # checks if the input is a sparse matrix (like from OneHotEncoder)
            X = X.toarray()  # convert sparse matrix to dense
        X = np.array(X, dtype=np.float32)
        device = next(self.parameters()).device
        # convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32,device=device)

        self.eval()
        # forward pass to get predictions
        with torch.no_grad():
            outputs = self(X_tensor)
            probabilities = torch.sigmoid(outputs).detach().cpu().numpy().flatten()

        return np.column_stack([1.0 - probabilities, probabilities])  # binary classification

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.decision_threshold).astype(np.int64)