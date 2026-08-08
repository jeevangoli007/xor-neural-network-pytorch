import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Step 1: Create XOR Dataset
# -------------------------------

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

# -------------------------------
# Step 2: Define Neural Network
# -------------------------------


class XORNet(nn.Module):

    def __init__(self):
        super(XORNet, self).__init__()

        self.hidden = nn.Linear(2, 4)   # Input -> Hidden
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1)   # Hidden -> Output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

# -------------------------------
# Step 3: Create Model
# -------------------------------


model = XORNet()

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Step 4: Training Loop
# -------------------------------

epochs = 5000

for epoch in range(epochs):

    # Forward Pass
    outputs = model(X)

    # Calculate Loss
    loss = criterion(outputs, y)

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    # Update Weights
    optimizer.step()

    # Print Loss Every 500 Epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

# -------------------------------
# Step 5: Testing
# -------------------------------

print("\nFinal Predictions:\n")

with torch.no_grad():

    predictions = model(X)

    predicted = (predictions >= 0.5).float()

    for i in range(len(X)):
        print(
            f"Input: {X[i].tolist()} | "
            f"Prediction: {predictions[i].item():.4f} | "
            f"Class: {int(predicted[i].item())} | "
            f"Expected: {int(y[i].item())}"
        )
