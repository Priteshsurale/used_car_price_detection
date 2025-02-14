import sys
import torch
import pandas as pd
from torch import nn


x_mean = torch.load('./model/x_mean.pt', weights_only=True)
y_mean = torch.load('./model/y_mean.pt', weights_only=True)
x_std = torch.load('./model/x_std.pt', weights_only=True)
y_std = torch.load('./model/y_std.pt', weights_only=True)


# print(x_mean)
# print(x_std)
# print()
# print(y_mean)
# print(y_std)


model = nn.Linear(2,1)

model.load_state_dict(torch.load('./model/model.pt', weights_only=True))

model.eval()

with torch.no_grad():
    x_data = torch.tensor([
        [5,10000],
        [2,10000],
        [5,20000]
    ], dtype=torch.float32)


    prediction = model((x_data - x_mean) / x_std)
    print(prediction * y_std + y_mean)



