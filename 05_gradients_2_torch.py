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

import torch

# f = w * x
# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediciton (manually)


def forward(x):
    return w*x

# loss (manually)


def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward()  # dl/dw

    with torch.no_grad():
        # update weights
        w -= learning_rate * w.grad

    if epoch % 10 == 0:
        print(
            f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}, grad = {w.grad:.5f}')
    # zero gradients
    w.grad.zero_()


print(f'Prediction after training: f(5) = {forward(5):.3f}')


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
