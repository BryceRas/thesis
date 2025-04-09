import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

class BoundaryNetwork(nn.Module):
    # Define the neural network architecture -- think is correct, may need to change neuron count
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(BoundaryNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim*4),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim * 4, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim),
            nn.Softplus()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # init.kaiming_normal_(layer.weight)  # Default initialization
                # init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                # init.constant_(layer.bias, 0.0)  # Default bias initialization

                # Apply Kaiming initialization with slight randomness
                init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                layer.weight.data += torch.randn_like(layer.weight) * 0.01  # Small noise
                
                # Initialize bias randomly (instead of always zero)
                init.uniform_(layer.bias, -0.1, 0.1)

    def forward(self, x):
        return self.model(x)

# Simulate an Ornstein-Uhlenbeck process -- think is correct
def simulate_ou_process(x0, mu, sigma, lam, dt, num_steps, num_paths):
    x = np.zeros((num_steps + 1, num_paths))
    x[0] = x0
    for t in range(1, num_steps + 1):
        dx = lam * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn(num_paths)
        x[t] = x[t - 1] + dx
    return x

# Define the CRRA utility function -- think is correct
def crra_utility(w, gamma=2):
    small_number = 1
    w[w <= 0] = small_number  # set 0s to small numbers to avoid issue
    if gamma == 1:
        return torch.log(w)
    else:
        return (w ** (1 - gamma) - 1) / (1 - gamma)

# Define the loss function -- think is correct
def loss_fn(terminal_wealth, starting_wealth, start_price, num_steps, model):
    return torch.mean(model.J(starting_wealth, start_price, num_steps) - crra_utility(terminal_wealth))

def update_wealth(W, pi_new, pi_old, price_change, beta):
    transaction_costs = beta * (pi_new - pi_old) ** (3/2)
    W_new = W + pi_new * price_change - transaction_costs
    return W_new