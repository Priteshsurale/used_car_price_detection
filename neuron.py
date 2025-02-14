import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt



df = pd.read_csv('./data/used_cars.csv')


price = df['price']
price = price.str.replace("$","")
price = price.str.replace(",","")
price = price.astype(int)

age = df['model_year'].max() - df['model_year']

milage = df['milage']
milage = milage.str.replace(",","")
milage = milage.str.replace(" mi.","")
milage = milage.astype(int)



X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)    
])


x_mean = X.mean(axis=0)
x_std = X.std(axis=0)

#  normalize the data 
X = (X- x_mean) / x_std


Y = torch.tensor(price, dtype=torch.float32).reshape((-1,1))
Y_mean = Y.mean()
Y_std = Y.std()

#  normalize the data 
Y = (Y - Y_mean) / Y_std

if not os.path.isdir('./model'):
    os.makedirs('./model', exist_ok=True)


torch.save(x_mean, './model/x_mean.pt')
torch.save(Y_mean, './model/y_mean.pt')
torch.save(x_std,'./model/x_std.pt')
torch.save(Y_std,'./model/y_std.pt')

    
model = nn.Linear(2,1)
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

losses = []
for i in range(2500):
    
    optimizer.zero_grad()
    prediction = model(X)
    loss =  loss_fun(prediction, Y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # if i %100 == 0 :
    #     print(loss)
    #     # print(model.weight)
    #     # print(model.bias)
        



print(model.state_dict())
torch.save(model.state_dict(), './model/model.pt')




# plt.plot(losses)
# plt.show()




# x_data = torch.tensor([
#     [5, 20000],
#     [2, 20000],
#     [1, 50000]
#     ], dtype=torch.float32)

# output = model((x_data - x_mean)/ x_std)
# print(output * Y_std + Y_mean)





"""
Purpose Of Normalization:
Normalizing output data brings values into a smaller, controlled range, making learning stable and weight updates smoother. This helps the model avoid large weight adjustments and prevents gradient issues, enabling the neuron to learn effectively.
"""