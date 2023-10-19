import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x, t):
        x = torch.cat([x, t], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    
    def generate(self, num_samples, logp):
        noise = torch.randn(num_samples, self.in_features).float()
        logp = logp.float()
        with torch.no_grad():
            generated_data = self.forward(noise, logp)
        return generated_data


input_dim = emb_smiles.shape[1]
hidden_dim = 256  # You can adjust this
output_dim = input_dim  # Assuming the output has the same dimensionality as the input

unet = MLP(input_dim + 320, hidden_dim, output_dim)  # +320 accounts for the time embedding
bfn_model = BFNContinuousData(unet, in_channels=input_dim)

model.load_state_dict(torch.load('./models/model_100.pth'))

model.eval()  # Set the model to evaluation mode
print(model)
with torch.no_grad():
    generated_data = model.generate(10)  # Assuming your model has a generate method

print(len(generated_data))
generated_data = generated_data.reshape(-1, 256).flatten().numpy()
scaled_emb_smiles_data =torch.from_numpy(np.load('Emb_Smiles/scaled_emb_smiles_data.npy')[:10]).flatten().numpy()

plt.hist(scaled_emb_smiles_data, bins=50, alpha=0.5, label='Original Data')
plt.hist(generated_data, bins=50, alpha=0.5, label='Generated Data')
plt.legend(loc='upper right')

# Save the figure before showing it
plt.savefig('histogram.png')

# Now show the plot
plt.show()