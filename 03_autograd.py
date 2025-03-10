import torch


# # Using  requires_grad ----------------------------------------------------------------
# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x + 2
# print(y)
# z = y * y * 2
# # z = z.mean()
# print(z)

# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

# z.backward(v)
# print(x.grad)


# # Three ways to remove grad ----------------------------------------------------------
# # 1
# x.requires_grad_(False)
# print(x)

# # 2
# y = x.detach()
# print(y)

# # 3
# with torch.no_grad():
#     y = x + 2
#     print(y)


# # Importance of setting grad to zero after each backward pass -----------------------
# weights = torch.ones(4, requires_grad=True)

# for epoch in range(3):
#     model_output = (weights*3).sum()

#     model_output.backward()

#     print(weights.grad)

#     weights.grad.zero_()


# # Optimiser -------------------------------------------------------------------------
# weights = torch.ones(4, requires_grad=True)

# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
