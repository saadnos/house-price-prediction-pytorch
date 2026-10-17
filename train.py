import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Model(nn.Module):
    def __init__(self, in_features=12, fc1=20, fc2=25, out_features=1):
        super().__init__()

        self.fc1 = nn.Linear(in_features,fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.out = nn.Linear(fc2,out_features)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

model = Model()

house_df = pd.read_csv('data.csv')

x = house_df.drop(['price', 'date','street','city','statezip','country'], axis=1).values.astype(np.float32)
y = house_df["price"].values.astype(np.float32).reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# Normalize input features
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

# Normalize house prices
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 20000
Losses = []

for i in range(epochs):

    y_pr = model.forward(x_train)

    Loss = criterion(y_pr, y_train)

    Losses.append(Loss.detach().numpy())

    if i % 10 == 0:
        print(f"{i}.) Losse : {Loss} \n")

    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

plt.plot(range(epochs), Losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()


# Test the model on unseen data
model.eval()

with torch.no_grad():
    predictions = model(x_test)

    # Convert predictions and actual values back to real prices
    predictions = y_scaler.inverse_transform(predictions.numpy())
    actual_prices = y_scaler.inverse_transform(y_test.numpy())

# Compare some predictions with real prices
for i in range(10):
    print(f"House {i + 1}")
    print(f"Predicted price: {predictions[i][0]:,.2f}")
    print(f"Actual price:    {actual_prices[i][0]:,.2f}")
    print()

torch.save(model.state_dict(), "house_model.pth")
