import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedNetwork, self).__init__()
        self.in_features = in_features  # Store in_features for later use
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)

        
    def forward(self, x,ema=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def generate(self, num_samples):
        # Replace with your generation logic
        # For example, you might generate random noise as input to the network:
        noise = torch.randn(num_samples, self.in_features)
        with torch.no_grad():
            generated_data = self.forward(noise)
        return generated_data

model = FullyConnectedNetwork(256, 256)  # Assuming this is the name of your model class
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