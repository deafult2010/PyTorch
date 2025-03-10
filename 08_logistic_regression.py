# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimiser
# 3) Training loop
#   - forward pass: compute predicition
#   - backward pass: gradients
#   - update weights
#   - iterate

import torch
import torch.nn as nn
import numpy as np
import random
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target


# X_numpy, y_numpy = datasets.make_regression(
#     n_samples=100, n_features=1, noise=20, random_state=1)

# X = torch.from_numpy(X_numpy.astype(np.float32))
# y = torch.from_numpy(y_numpy.astype(np.float32))
# y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(LogisticRegression, self).__init__()
        # define layers
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

# # plot
# predicted = model(X).detach()
# print(predicted)
# print(X_numpy)
# print(predicted[0]/X_numpy[0])
# plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b')
# plt.show()
