import torch
import numpy as np

# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# print(x)
# print(y)
# z = x+y
# z = torch.add(x, y)
# print(z)


# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x+y
#     z = z.to("cpu")
#     print(x)
#     print(y)
#     print(z)


# Step 1: Initialize matrices `a` and `b` with fixed values
a = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32,
                 requires_grad=True)  # 3x2 matrix
b = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32,
                 requires_grad=True)  # 2x2 matrix

# Step 2: Perform matrix multiplication
c = torch.matmul(a, b)  # Resulting in a 3x2 matrix

# Print the result of the matrix multiplication
print("Matrix c (result of a @ b):")
print(c)

# Step 3: Enable gradients for `a` and `b`
a.requires_grad_()

# Step 4: Perform backpropagation to compute the gradients of `a`
c.backward(torch.ones_like(c))

# Print the gradients of `a`
print("Gradient of a:")
print(a.grad)
