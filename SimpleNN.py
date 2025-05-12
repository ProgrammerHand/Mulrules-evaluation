from lux.lux import LUX
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

    def fit(self,X,y):
      # Convert data to tensors
      X_train_tensor = torch.tensor(X, dtype=torch.float32)
      y_train_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

      # Loss and optimizer
      pos_weight = torch.tensor([(y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()])
      criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
      # criterion = nn.BCELoss()
      optimizer = optim.Adam(self.parameters(), lr=0.0001)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

      best_loss = float('inf')

      print(f"First few samples: {X[:5]}")
      print(f"Target distributions: {np.bincount(y.astype(np.int64))}")
      epochs = 1000
      # Training loop
      for epoch in range(epochs):
          optimizer.zero_grad()
          outputs = self(X_train_tensor)

          if epoch == 0:
              print(f"Initial outputs: {outputs[:5]}")
              print(f"Initial targets: {y_train_tensor[:5]}")

          loss = criterion(outputs, y_train_tensor)
          loss.backward()

          if epoch % 10 == 0:
              with torch.no_grad():
                  grad_norm = sum(p.grad.norm().item() for p in self.parameters() if p.grad is not None)
                  print(f"Gradient norm: {grad_norm}")

                  predictions = (torch.sigmoid(outputs) >= 0.5).int()
                  accuracy = (predictions == y_train_tensor).float().mean()
                  print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

          optimizer.step()
          scheduler.step(loss)

    def predict_proba(self, X):
        # Ensure input is dense (convert if sparse)
        if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
            X = X.toarray()  # Convert sparse matrix to dense
        X = np.array(X, dtype=np.float32)
        # Convert to tensor if necessary
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Perform the forward pass to get predictions
        with torch.no_grad():
            outputs = self(X_tensor)
            probabilities = outputs.numpy().flatten()

        # Convert to probabilities (binary classification)
        # probabilities = outputs.numpy()  # Convert to numpy array

        return np.column_stack([1 - probabilities, probabilities])  # For binary classification

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(np.int64)
        # Ensure input is dense (convert if sparse)
        # if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
        #     X = X.toarray()  # Convert sparse matrix to dense
        # X = np.array(X, dtype=np.float32)
        # # Convert to tensor if necessary
        # X_tensor = torch.tensor(X, dtype=torch.float32)
        #
        # # Perform the forward pass to get predictions
        # with torch.no_grad():
        #     outputs = self(X_tensor)
        #
        # # Classify based on the output probability (threshold of 0.5)
        # predictions = (outputs >= 0.5).int().numpy().flatten().astype(np.int64) # Binary classification: 0 or 1
        # # if predictions.numel() == 1:
        # #     return np.array([int(predictions.item())])
        # # else:
        # #     return predictions.numpy().astype(int).flatten()
        # return predictions