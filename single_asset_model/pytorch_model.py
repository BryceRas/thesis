import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pytorch_helper import BoundaryNetwork, loss_fn
from analytic_model import analytic_solution

import os

def train(lam, mu, sigma, x0, ticker, run_number, beta,linear=False):
    ## Hyperparameters
    num_epochs = 2000
    learning_rate = 1e-4
    hidden_dim = 64
    input_dim = 4  # Asset price, time
    num_steps = 250  # Simulation steps
    num_paths = 10000  # Simulation paths
    ra = 2  # Risk aversion
    T = 3.0  # Time to maturity
    dt = T / num_steps
    # beta = 1e-4  # Transaction cost coefficient
    # beta = 0.03
    r = 0.05  # Risk-free rate
    w0 = 1000  # Initial wealth
    h = dt
    var = sigma**2 * (1 - np.exp(-2 * lam * h)) / (2 * lam)

    # Set default device to GPU
    torch.set_default_device("cuda")
    torch.set_printoptions(sci_mode=False)

    # Initialize the analytic solution
    analytic_model = analytic_solution(ra, r, lam, mu, sigma)

    # Initialize models, optimizers

    upper_net = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    lower_net = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    optimizer = optim.AdamW(list(upper_net.parameters()) + list(lower_net.parameters()), lr=learning_rate,weight_decay=1e-4)
    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # plotting data
    average_wealths = []
    losses = []
    final_wealths = []
    # lowers= []
    # uppers = []

    for epoch in range(num_epochs+1):
        # Simulate price paths
        wealth = torch.ones((num_paths, 1), device='cuda') * w0
        x = torch.ones((num_paths, 1), device='cuda') * x0
        pi_minus = torch.zeros_like(wealth, device='cuda')  # Starting with no position
        y = torch.ones((num_paths, 1), device='cuda') * w0  # All wealth in riskless
        total_costs = torch.zeros_like(wealth, device='cuda')  # Initialize total costs
        t = 0
        for _ in range(num_steps):
            x = x.to(torch.float32)
            wealth = wealth.to(torch.float32)
            pi_minus = pi_minus.to(torch.float32)
            time_left = (T-t)*torch.ones((num_paths, 1), device='cuda')
            inputs = torch.cat([x, wealth, pi_minus,time_left], dim=1)
            # inputs = torch.cat([x], dim=1)

            no_cost_position = torch.clamp(analytic_model.optimal_nocost((T-t), x), -1.0, 1.0)
            
            # Predict boundaries
            upper_offset = upper_net(inputs)
            lower_offset = lower_net(inputs)
            upper_boundary = torch.clamp(no_cost_position + upper_offset, -1.0, 1.0)
            lower_boundary = torch.clamp(no_cost_position - lower_offset, -1.0, 1.0)
            pi_plus = torch.clamp(pi_minus, lower_boundary, upper_boundary)

            t = t + h

            # Simulate OU process
            dx = (mu - mu * np.exp(-lam * h) + np.exp(-lam * h) * x - x + torch.normal(torch.tensor(0.0, device='cuda'), 
                                        torch.sqrt(torch.tensor(var, device='cuda')), (num_paths, 1), device='cuda'))
            dy = (np.exp(r * h) - 1)
            if linear:
                c = beta * wealth * torch.abs(pi_plus - pi_minus)
            else:
                c = beta * (torch.pow(torch.abs(wealth * torch.abs(pi_plus - pi_minus)),1.5))
            # c = beta * (wealth * (torch.abs(pi_plus - pi_minus)))
            total_costs = total_costs + c
            dw = (pi_plus * wealth / x) * dx + (1 - pi_plus) * wealth * dy - c

            wealth_in_stock = wealth * pi_plus + (pi_plus * wealth / x) * dx
            
            wealth = wealth + dw
            x = x + dx
            y = wealth - wealth_in_stock
            pi_minus = wealth_in_stock/wealth

        # Compute loss
        loss = loss_fn(wealth, w0, x0, T, analytic_model)

        losses.append(loss.item())
        average_wealths.append(torch.mean(wealth).item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(upper_net.parameters()) + list(lower_net.parameters()),  # Parameters from both models
            max_norm=5  # Gradient norm threshold
        )
        
        optimizer.step()

        # Step the scheduler for learning rate decay
        scheduler.step()
        # Print progress
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # print(pi_plus)
            print((
                f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.10f}, LR: {current_lr:.6f}, "
                f"Average Wealth: {round(torch.mean(wealth).item(), 0)}, "
                f"Average total costs paid: {round(torch.mean(total_costs).item(), 0)}"
            ))

    print('Training finished.')

    fig,ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Wealth', color='b')
    ax.set_title(f'Average End Portfolio Wealth for {ticker} Over Model Training')
   
    ax.plot(average_wealths, color='b')
    # ax2.plot(uppers, color='r',label='upper offset')
    # ax2.plot(lowers, color='g', label = 'lower offset')
    plt.savefig(f'training_graphs/training_plot_{ticker}_model_{run_number}.png')
    plt.close()
    return upper_net, lower_net
