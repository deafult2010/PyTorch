# 1 ---------------------------------------------------------------------------------
# Prediction: Manually
# Gradient Computation: Manually
# Loss Computation: Manually
# Param Updates: Manually


# 2 ---------------------------------------------------------------------------------
# Prediction: Manually
# Gradient Computation: Autograd
# Loss Computation: Manually
# Param Updates: Manually


# 3 ---------------------------------------------------------------------------------
# Prediction: Manually
# Gradient Computation: Autograd
# Loss Computation: PyTorch Loss
# Param Updates: PyTorch Optimizer


# 4 ---------------------------------------------------------------------------------
# Prediction: PyTorch Model
# Gradient Computation: Autograd
# Loss Computation: PyTorch Loss
# Param Updates: PyTorch Optimizer


# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimiser
# 3) Training loop
#   - forward pass: compute predicition
#   - backward pass: gradients
#   - update weights
#   - iterate

import torch
import torch.nn as nn

# f = w * x
# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5],  dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward()  # dl/dw

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(
            f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
