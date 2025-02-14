import pandas as pd
import sys, os
import torch
from torch import nn


df = pd.read_csv('./data/used_cars.csv')


price = df['price']
price = price.str.replace("$","")
price = price.str.replace(",","")
price = price.astype(int)

age = df['model_year'].max() - df['model_year']

accident_free = df["accident"] == "None reported"
accident_free = accident_free.astype(int)


milage = df['milage']
milage = milage.str.replace(",","")
milage = milage.str.replace(" mi.","")
milage = milage.astype(int)



X = torch.column_stack([
    torch.tensor(accident_free, dtype=torch.float32), 
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])


x_mean = X.mean(axis=0)
x_std = X.std(axis=0)


"""
z score normalization z = (X – μ) / σ,

X is the value of the data point
μ is the mean of the dataset
σ is the standard deviation of the dataset
""" 
#  normalize the data 
X = (X- x_mean) / x_std


Y = torch.tensor(price, dtype=torch.float32).reshape((-1,1))
Y_mean = Y.mean()
Y_std = Y.std()


#  normalize the data 
Y = (Y - Y_mean) / Y_std



    
model = nn.Linear(3,1)
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

losses = []
for i in range(2500):
    
    optimizer.zero_grad()
    prediction = model(X)
    loss =  loss_fun(prediction, Y)
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0: 
        print(loss)
        # print(model.weight)
        # print(model.bias)
        
        
        

x_data = torch.tensor([
    [1, 5, 10000],
    [1, 2, 10000],
    [1, 5, 20000]
], dtype=torch.float32)

prediction = model((x_data - x_mean )/ x_std)
print(prediction * Y_std + Y_mean)


        
x_data_accident = torch.tensor([
    [0, 5, 10000],
    [0, 2, 10000],
    [0, 5, 20000]
], dtype=torch.float32)

prediction_accident = model((x_data_accident - x_mean )/ x_std)
print(prediction_accident * Y_std + Y_mean)



