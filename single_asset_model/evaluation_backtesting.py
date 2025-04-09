import pandas as pd
import numpy as np
import torch

from analytic_model import analytic_solution

# load data for evaluation or testing
def load_data(stock_ticker, testing):
    df = pd.read_csv(f"../beta/{stock_ticker.upper()}.csv")
    df.rename(columns={df.columns[2]: 'ETF'}, inplace=True)
    df = df[['date','ETF','beta','price']]
    df['date']= pd.to_datetime(df['date'])
    if testing:
        df = df[df['date'].dt.year.isin([2019, 2020])]
    else: df = df[df['date'].dt.year > 2020]
    return df


def backtest(ticker, upper_model, lower_model, coeffs, beta, use_nn=False, test=False, linear=False):
    lam, mu, sigma = coeffs['lambda'], coeffs['mu'], coeffs['sigma']
    
    ra, r = 2, 0.05
    wealth = 1000
    
    df = load_data(ticker, test)
    T = df['date'].dt.year.nunique()
    trading_days = df['date'].nunique()
    dt = T / trading_days
    pi_minus = 0
    Beta = 0
    wealths = []
    portfolio_balance = []
    trading_costs_paid = []
    carry_costs = [0]

    # Initialize the analytic solution
    analytic_model = analytic_solution(ra, r, lam, mu, sigma)

    for day in range(trading_days):
        if day != 0:
            dBeta = df['beta'].iloc[day] - Beta
            cc = dBeta * df['ETF'].iloc[day] * (pi_plus * wealth / x)
            cc = 0 # Calculate the carry costs
            carry_costs.append(cc)
            dx = df['price'].iloc[day] - x
            dy = np.exp(r*dt)-1
            dw = (pi_plus * wealth / x) * dx + (1-pi_plus) * dy * wealth - c - cc
            share_count = wealth * pi_plus / x
            wealth = wealth + dw
            pi_minus = (share_count * (x+dx)) / wealth
        
        x = df['price'].iloc[day]
        Beta = df['beta'].iloc[day]

        # Calculate the upper and lower bounds
        if use_nn:
            upper_offset = upper_model(torch.tensor([[x, wealth, pi_minus, T-day*dt]],dtype=torch.float).to('cuda')).item()
            lower_offset = lower_model(torch.tensor([[x, wealth, pi_minus, T-day*dt]],dtype=torch.float).to('cuda')).item()
        else: upper_offset, lower_offset = 0, 0
        no_cost_position = np.clip(analytic_model.optimal_nocost((T-day*dt), x), -1.0, 1.0)
        upper_boundary = np.clip(no_cost_position + upper_offset, -1.0, 1.0)
        lower_boundary = np.clip(no_cost_position - lower_offset, -1.0, 1.0)
        pi_plus = np.clip(pi_minus, lower_boundary, upper_boundary)

        if linear == False: c = beta * (wealth * np.abs(pi_plus-pi_minus))**1.5
        else: c = beta * (wealth * np.abs(pi_plus-pi_minus))
        trading_costs_paid.append(c)

        wealths.append(wealth)
        portfolio_balance.append(pi_plus)
    temp = df.copy()
    temp['wealth'] = wealths
    temp['portfolio_balance'] = portfolio_balance
    temp['trading_costs_paid'] = trading_costs_paid
    temp['carry_costs'] = carry_costs
    temp.drop(columns=['ETF','beta'], inplace=True)
    return temp
