import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. DATA — XOR problem using NumPy
# ─────────────────────────────────────────
# XOR truth table:
#   0 XOR 0 = 0
#   0 XOR 1 = 1
#   1 XOR 0 = 1
#   1 XOR 1 = 0

X_np = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert NumPy arrays to PyTorch tensors
X = torch.tensor(X_np)
y = torch.tensor(y_np)

# ─────────────────────────────────────────
# 2. MODEL — Define the neural network
# ─────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Input layer (2 inputs) → Hidden layer (8 neurons) → Output layer (1 output)
        self.fc1 = nn.Linear(2, 8)   # First layer
        self.fc2 = nn.Linear(8, 8)   # Hidden layer
        self.fc3 = nn.Linear(8, 1)   # Output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()   # Squishes output between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = NeuralNet()
print("Model architecture:")
print(model)
print()

# ─────────────────────────────────────────
# 3. TRAINING SETUP
# ─────────────────────────────────────────
criterion = nn.BCELoss()               # Binary Cross Entropy loss (good for 0/1 outputs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

epochs = 2000   # How many times we loop through the data
losses = []     # Track loss over time so we can plot it

# ─────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────
print("Training...")
for epoch in range(epochs):
    # Forward pass — run data through the model
    predictions = model(X)

    # Calculate how wrong the model is
    loss = criterion(predictions, y)
    losses.append(loss.item())

    # Backward pass — figure out how to improve
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update weights

    # Print progress every 200 epochs
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs}  |  Loss: {loss.item():.4f}")

print("\nTraining complete!")

# ─────────────────────────────────────────
# 5. RESULTS
# ─────────────────────────────────────────
print("\n--- Predictions vs Expected ---")
with torch.no_grad():   # No need to track gradients during testing
    output = model(X)
    predicted = (output >= 0.5).float()   # Round to 0 or 1

    for i in range(len(X_np)):
        inputs = X_np[i]
        expected = int(y_np[i][0])
        pred = int(predicted[i][0].item())
        confidence = output[i][0].item()
        status = "✓" if pred == expected else "✗"
        print(f"  {int(inputs[0])} XOR {int(inputs[1])} = {pred}  (expected {expected}, confidence: {confidence:.2f})  {status}")

# ─────────────────────────────────────────
# 6. PLOT — Visualize the training loss
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(losses, color='royalblue', linewidth=1.5)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()
print("\nLoss chart saved as training_loss.png")