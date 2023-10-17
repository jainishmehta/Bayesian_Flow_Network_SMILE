import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.nn.functional as F
import torch

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        return self.fc(x)

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

# Assuming regressor is your trained regressor from step 1
# and model is your ConditionalFullyConnectedNetwork
model = ConditionalFullyConnectedNetwork(256, 256)
regressor = Regressor()

# Ensure models are in evaluation mode

model = ConditionalFullyConnectedNetwork(256, 256)
model.load_state_dict(torch.load('./conditional_models/model.pth'))
model.eval()
regressor.eval()

# Step 1: Generate Data
logp_control = torch.linspace(-5, 5, 100)  # Control logP values
generated_data = model.generate(100, logp_control)

# Step 2: Infer logP
# Convert generated_data to a format that Regressor can handle
generated_data = generated_data.view(100, -1)
with torch.no_grad():  # Ensure no_grad context for inference
    inferred_logp = regressor(generated_data)

# Step 3: Compare
mse = mean_squared_error(logp_control.numpy(), inferred_logp.detach().numpy())
print(f'Mean Squared Error between control and inferred logP: {mse}')

# Step 4: Visualization
plt.scatter(logp_control.numpy(), inferred_logp.detach().numpy())
plt.xlabel('Control logP')
plt.ylabel('Inferred logP')
plt.title('Control vs Inferred logP')
plt.plot([-1, 1], [-1, 1], color='red')  # Line y=x for reference
plt.savefig('control_inferred_logp.png')

# Now show the plot
plt.show()
