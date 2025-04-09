import pandas as pd
import numpy as np
import torch

from analytic_model import analytic_solution

# load data for evaluation
def load_data(stock_tickers, testing):
    prices = []
    stock_betas = []
    etf_prices = []
    for ticker in stock_tickers:
        np.set_printoptions(suppress=True, precision=5)
        df = pd.read_csv(f'../beta/{ticker}.csv')
        df.rename(columns={df.columns[2]: 'ETF'}, inplace=True)
        df = df[['date','ETF','beta','price']]
        df['date'] = pd.to_datetime(df['date'])
        if testing:
            df = df[df['date'].dt.year.isin([2019, 2020])]
        else: df = df[df['date'].dt.year > 2020]

        prices.append(df[['price']].rename(columns={'price': ticker}))
        stock_betas.append(df[['beta']].rename(columns={'beta': ticker}))
        etf_prices.append(df[['ETF']].rename(columns={'ETF': ticker}))
    prices = pd.concat(prices, axis=1)
    stock_betas = pd.concat(stock_betas, axis=1)
    etf_prices = pd.concat(etf_prices, axis=1)
    dates = df[['date']]
    
    return prices, stock_betas, dates, etf_prices


def backtest(tickers, upper_model, lower_model, coeffs, beta, use_nn=False,test=False, linear=False):
    lam, mu, SigmaOU = coeffs['lambda'], coeffs['mu'], coeffs['SigmaOU']
    n = len(lam)
    ra, r = 2, 0.05
    wealth = 1000
    
    prices, betas, dates, etf_prices = load_data(tickers, test)
    T = dates['date'].dt.year.nunique()
    trading_days = dates['date'].nunique()
    dt = T / trading_days
    pi_minus = np.zeros(n)
    Beta = 0
    wealths = []
    portfolio_balance = []
    trading_costs_paid = []
    carry_costs = [0]

    # Initialize the analytic solution
    analytic_model = analytic_solution(n, ra, r, lam, mu, SigmaOU,T, trading_days) 

    for day in range(trading_days):
        if day != 0:
            dBeta = betas.iloc[day].to_numpy() - Beta
            cc = np.sum(dBeta * etf_prices.iloc[day] * (pi_plus * wealth / x))
            # cc = 0 # Calculate the carry costs
            carry_costs.append(cc)
            dx = prices.iloc[day].to_numpy() - x
            dy = np.exp(r*dt)-1
            dw = (pi_plus * wealth / x) @ dx + (1-np.sum(pi_plus)) * dy * wealth - c - cc
            share_count = wealth * pi_plus / x
            wealth = wealth + dw
            pi_minus = (share_count * (x+dx)) / wealth
        
        x = prices.iloc[day].to_numpy()
        Beta = betas.iloc[day].to_numpy()

        # Calculate the upper and lower bounds
        if use_nn:
            inputs = torch.cat([
                torch.from_numpy(x).float().to('cuda'),
                torch.tensor([wealth], dtype=torch.float32,device='cuda'),
                torch.from_numpy(pi_minus).float().to('cuda'), 
                # torch.tensor([wealth, T - day * dt], dtype=torch.float32,device='cuda')
            ]).float().to("cuda").unsqueeze(0)
            upper_offset = upper_model(inputs).cpu().detach().numpy()[0]
            lower_offset = lower_model(inputs).cpu().detach().numpy()[0]
        else: upper_offset, lower_offset = 0, 0
        no_cost_position = np.clip(analytic_model.no_cost_optimal(x, day), -1.0, 1.0)
        upper_boundary = np.clip(no_cost_position + upper_offset, -1.0, 1.0)
        lower_boundary = np.clip(no_cost_position - lower_offset, -1.0, 1.0)
        pi_plus = np.clip(pi_minus, lower_boundary, upper_boundary)
        
        if linear == False: c = beta * np.sum(np.power(wealth * np.abs(pi_plus-pi_minus),1.5))
        else: c = beta * np.sum(wealth * np.abs(pi_plus-pi_minus))
        trading_costs_paid.append(c)

        wealths.append(wealth)
        portfolio_balance.append(pi_plus)
    prices = prices.add_suffix("_price")
    betas = betas.add_suffix("_beta")
    etf_prices = etf_prices.add_suffix("_ETFPrice")
    temp = pd.concat([dates,prices, betas,etf_prices], axis=1)
    temp['wealth'] = wealths
    temp['portfolio_balance'] = portfolio_balance
    temp['trading_costs_paid'] = trading_costs_paid
    temp['carry_costs'] = carry_costs
    # temp.drop(columns=['ETF','beta'], inplace=True)
    return temp
