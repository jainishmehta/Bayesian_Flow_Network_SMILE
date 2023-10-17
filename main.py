import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
with open('bfn_exercise.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract 'emb_smiles' and 'logp'
X = np.array([row['emb_smiles'] for row in data])
y = np.array([row['logp'] for row in data])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network regressor
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        return self.fc(x)

# Train the regressor
regressor = Regressor()
optimizer = optim.Adam(regressor.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = regressor(torch.Tensor(X_train))
    loss = criterion(outputs.view(-1), torch.Tensor(y_train))
    loss.backward()
    optimizer.step()
"""
# Test the regressor on the test set
test_predictions = regressor(torch.Tensor(X_test)).detach().numpy()
mse = mean_squared_error(y_test, test_predictions)
print("Mean Squared Error:", mse)
"""
def scale_data(data):
    min_val = data.min()
    max_val = data.max()
    scaled_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return scaled_data
    
emb_smiles_data = X
logp_data = y
# Scale 'emb_smiles' data to (-1, 1) range
scaled_emb_smiles_data = scale_data(emb_smiles_data)
logp_data = scale_data(logp_data)

print(scaled_emb_smiles_data)
print(type(scaled_emb_smiles_data))
print(np.shape(scaled_emb_smiles_data))
# ... Your code for scaling 'emb_smiles' data ...
# Save the scaled 'emb_smiles' data as a NumPy array
np.save("scaled_emb_smiles_data.npy", scaled_emb_smiles_data)
np.save("scaled_logp_data", logp_data)
