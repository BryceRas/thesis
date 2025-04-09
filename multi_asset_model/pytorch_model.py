import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from analytic_model import analytic_solution
from pytorch_helper import BoundaryNetwork, loss_fn,crra_utility
from evaluation_backtesting import backtest

def train(coeffs, model_number, run_number, beta,  linear=False):
    # set asset coeffs
    lambd = np.array(coeffs['lambda'])
    mu = np.array(coeffs['mu'])
    # sigma = np.array(coeffs['sigma'])
    # corr = np.array(coeffs['corr'])
    SigmaOU = np.array(coeffs['SigmaOU'])
    x0 = torch.tensor(coeffs['x0'], device='cuda', dtype=torch.float32)
    n = len(lambd)

    ## Hyperparameters
    num_epochs = 5000
    learning_rate = 1e-4
    # hidden_dim = 196
    hidden_dim = 64

    input_dim = 2 * n +1 # Asset price and balance for each asset, timeleft, total wealth
    num_steps = 150  # Simulation steps
    num_paths = 20000  # Simulation paths
    ra = 2  # Risk aversion
    T = 3.0  # Time to maturity
    dt = T / num_steps
    # beta = 1e-4  # Transaction cost coefficient
    # beta = 0.03
    r = 0.05  # Risk-free rate
    w0 = 1000  # Initial wealth
    h = dt
    
    starting_values = torch.tensor([1000, 2000, 5000, 10000, 20000], device='cuda', dtype=torch.float32)
    w0s = starting_values.repeat(num_paths // len(starting_values) + 1)[:num_paths].view(num_paths, 1)
    

    average_wealths = []
    losses = []
    final_wealths = []

    # Set default device to GPU
    torch.set_default_device("cuda")
    torch.set_printoptions(sci_mode=False)

    # Initialize the analytic solution
    analytic_model = analytic_solution(n, ra, r, lambd, mu, SigmaOU, T, num_steps)

    # Initialize models, optimizers

    upper_net = BoundaryNetwork(input_dim, hidden_dim, output_dim=n).to('cuda')
    lower_net = BoundaryNetwork(input_dim, hidden_dim, output_dim=n).to('cuda')
    optimizer = optim.AdamW(list(upper_net.parameters()) + list(lower_net.parameters()), lr=learning_rate,weight_decay=1e-4)
    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    lambd = torch.tensor(lambd, device='cuda', dtype=torch.float32)
    mu = torch.tensor(mu, device='cuda', dtype=torch.float32) 
    # sigma = torch.tensor(sigma, device='cuda', dtype=torch.float32)
    # corr = torch.tensor(corr, device='cuda', dtype=torch.float32)
    SigmaOU = torch.tensor(SigmaOU, device='cuda', dtype=torch.float32)  
    x0 = torch.tensor(coeffs['x0'], device='cuda', dtype=torch.float32)

    # dynamics constants
    sq = torch.sqrt((1-torch.exp(-2*h*lambd))/ (2*lambd))
    for epoch in range(num_epochs+1):
        # Simulate price paths
        # wealth = torch.ones((num_paths, 1), device='cuda') * w0
        wealth = w0s.clone()
        x = (torch.ones((1, n), device='cuda') * x0).repeat(num_paths, 1)
        pi_minus = torch.zeros((num_paths, n), device='cuda')  # Starting with no position
        y = torch.ones((num_paths, 1), device='cuda') * w0  # All wealth in riskless
        total_costs = torch.zeros_like(wealth, device='cuda')  # Initialize total costs
        t = 0

        for _ in range(num_steps):
            # x = x.to(torch.float32)
            # wealth = wealth.to(torch.float32)
            # pi_minus = pi_minus.to(torch.float32)
            time_left = (T-t)*torch.ones((num_paths, 1), device='cuda')
            inputs = torch.cat([x, wealth, pi_minus], dim=1)
            # inputs = torch.cat([x, pi_minus], dim=1)
            
            no_cost_position = torch.clamp(analytic_model.no_cost_optimal_matrix_torch(x, _), -1.0, 1.0)
            
            # Predict boundaries
            upper_offset = upper_net(inputs)
            lower_offset = lower_net(inputs)
            upper_boundary = torch.clamp(no_cost_position + upper_offset, -1.0, 1.0)
            lower_boundary = torch.clamp(no_cost_position - lower_offset, -1.0, 1.0)
            pi_plus = torch.clamp(pi_minus, lower_boundary, upper_boundary)
            
            t = t + h
            # break
            # Simulate OU process
            N = torch.randn((num_paths, n), device='cuda')
            dx = (mu - x) * (1-torch.exp(-h* lambd)) + sq* N @ SigmaOU.T
            dy = (np.exp(r * h) - 1) # dy/y
            if linear == False:
                c = beta * (wealth * (pi_plus - pi_minus).abs()).pow(1.5).sum(dim=1, keepdim=True)
            else: 
                c = beta * (wealth.abs() * (pi_plus - pi_minus).abs()).sum(dim=1, keepdim=True)
            # c = beta * (wealth * (pi_plus - pi_minus).abs()).sum(dim=1, keepdim=True)
            
            total_costs += c
            dw = torch.sum((dx/x)*pi_plus,dim=1, keepdim=True) * wealth + (1-torch.sum(pi_plus,dim=1, keepdim=True))*dy*wealth - c
            wealth_in_stock = wealth * pi_plus + (pi_plus * wealth / x) * dx
            wealth = wealth + dw
            x = x + dx
            pi_minus = wealth_in_stock/wealth
            
        # Compute loss
        # loss = loss_fn(wealth, w0, x0, num_steps, analytic_model)
        loss = -torch.mean(crra_utility((wealth/w0s)*1000))

        losses.append(loss.item())
        average_wealths.append(torch.mean(wealth/w0s*1000).item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(upper_net.parameters()) + list(lower_net.parameters()),  # Parameters from both models
            max_norm=5  # Gradient norm threshold
        )
        optimizer.step()

        # Step the scheduler for learning rate decay
        scheduler.step(loss.item())
        # Print progress
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # print(pi_plus)
            print((
                f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.10f}, LR: {current_lr:.6f}, "
                f"Average Wealth: {round(torch.mean(wealth/w0s*1000).item(), 0)}, "
                f"Average total costs paid: {round(torch.mean(total_costs/w0s*1000).item(), 0)}"
            ))

    print('Training finished.')

    tickers = coeffs['tickers']
    fig,ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Wealth', color='b')
    ax.set_title(f'Average End Portfolio Wealth for {tickers} Over Model Training')
   
    ax.plot(average_wealths, color='b')
    plt.savefig(f'training_graphs/training_plot_{tickers}_model_{model_number}_{run_number}.png')
    plt.close()
            
    return upper_net, lower_net