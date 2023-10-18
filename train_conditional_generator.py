import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFullyConnectedNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConditionalFullyConnectedNetwork, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features + 1, 128)  # +1 for the logp value
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x, logp):
        # Concatenate logp with x along the feature dimension
        x = torch.cat((x, logp.unsqueeze(-1)), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def generate(self, num_samples, logp):
        noise = torch.randn(num_samples, self.in_features).float()
        logp = logp.float()
        with torch.no_grad():
            generated_data = self.forward(noise, logp)
        return generated_data

model = ConditionalFullyConnectedNetwork(256, 256)
model.load_state_dict(torch.load('./conditional_models/model.pth'))
model.eval()
logp_values = torch.from_numpy(np.load('scaled_logp_data.npy')[:1000])  # Ensure logp_values is a tensor
emb_smiles_values = torch.from_numpy(np.load('./Emb_Smiles/scaled_emb_smiles_data.npy')[:1000])  # Ensure logp_values is a tensor

# Assume logp_values is a tensor of logP values for which you want to generate data
with torch.no_grad():  # Disable gradient computation
    generated_data = model.generate(1000, logp_values)  # Use the generate method instead of forward

# This could include comparing distributions, using domain-specific metrics, etc.
# Example: Compute Mean Squared Error (MSE) between generated data and real data
mse_loss = nn.MSELoss()(generated_data, emb_smiles_values )
print(f'MSE Loss: {mse_loss.item()}')
import matplotlib.pyplot as plt

generated_data = generated_data.reshape(-1, 256).flatten().numpy()
scaled_emb_smiles_data =torch.from_numpy(np.load('Emb_Smiles/scaled_emb_smiles_data.npy')[:1000]).flatten().numpy()

plt.hist(scaled_emb_smiles_data, bins=50, alpha=0.5, label='Original Data')
plt.hist(generated_data, bins=50, alpha=0.5, label='Generated Data')
plt.legend(loc='upper right')

# Save the figure before showing it
plt.savefig('histogram_conditional.png')

# Now show the plot
plt.show()
