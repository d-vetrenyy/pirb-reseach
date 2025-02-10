import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# Create a dataset
num_samples = 100
X = np.random.rand(num_samples, 2)  # 2D input
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple binary classification

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to hidden layer
        self.fc2 = nn.Linear(4, 2)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)               # Output layer
        return x

# Instantiate the model, define the loss function and the optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)

    # Compute the loss
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualizing the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    _, Z = torch.max(Z, 1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.detach().numpy(), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(model, X, y)
