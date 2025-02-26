import torch
import torch.nn as nn
import numpy as np
import random

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# # Methods:
# torch.save(arg, PATH)
# torch.load(PATH)
# model.load_state_dict(arg)


# ### COMPLETE MODEL ####
# torch.save(model, PATH)

# # model class must be definged somewhere
# model = torch.load(PATH)
# model.eval()


# ### STATE DICT ###
# torch.save(model.state_dict(), PATH)

# # model must be created again with parameters
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.Linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)

# train your model...


# for param in model.parameters():
#     print(param)

# FILE = "model.pth"
# # torch.save(model.state_dict(), FILE)

# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# model.eval()

# for param in loaded_model.parameters():
#     print(param)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
print(model.state_dict())

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

# torch.save(checkpoint, "checkpoint.pth")
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict())


# # Save on GPU; Load on CPU:
# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), PATH)

# device = torch.device("cpu")
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, map_location=device))


# # Save on GPU; Load on GPU:
# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), PATH)

# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.to(device)


# # Save on CPU; Load on GPU:
# torch.save(model.state_dict(), PATH)

# device = torch.device("cuda")
# model = Model(*args, **kwargs)
# # Chose whatever GPU you want
# model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# model.to(device)
