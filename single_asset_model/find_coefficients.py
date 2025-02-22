import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

def collect_values(stock_tickers: list[str], end_year: int) -> dict:
    coeffs = {}
    for ticker in stock_tickers:
        np.set_printoptions(suppress=True, precision=5)

        df = pd.read_csv(f'beta/{ticker}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year <= end_year]
        x0 = df['price'].iloc[-1]
        x =df['price'][:-1]
        y = df['price'][1:]

        def neg_log_likelihood(params):
            lambd, mu, sigma = params
            y_pred = lambd * (mu - x)+x
            return -np.sum(norm.logpdf(y, y_pred, sigma))

        initial_guess = [10, 30, 10]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        lambd, mu, sigma = result.x *np.array([252, 1, np.sqrt(252)])
        coeffs[ticker] = {'lambda': lambd, 'mu': mu, 'sigma': sigma, 'x0': x0}
        print(f'Lambda = {lambd} mu = {mu} sigma = {sigma}')
    
    return coeffs