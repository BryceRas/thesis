import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def collect_values(stock_tickers: list[str], end_year: int) -> dict:
    coeffs = {}
    coeffs['tickers'] = stock_tickers
    coeffs['end_year'] = end_year

    df = None
    for t in stock_tickers:
        data = pd.read_csv(f'../beta/{t}.csv')
        data = data[['date','price']]
        data['date'] = pd.to_datetime(data['date'])
        data = data[data['date'].dt.year <= 2020]
        data.rename(columns={'price': t}, inplace=True)
        if df is None:
            df = data
        else:
            df = pd.merge(df, data, on='date')
    df.index = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'])

    X = df.to_numpy()  # Replace with real dataset
    T, n = X.shape  # Number of time steps, number of assets
    dt = 1  # Time step (adjust based on data)

    # **Negative Log-Likelihood Function**
    def neg_log_likelihood(params):
        """
        Compute the negative log-likelihood of the OU process with structured drift μX(X) = Λ(M - X).
        """
        M = params[:n]  # Extract M (vector)
        lambda_diag = params[n:2*n]  # Extract diagonal elements of Λ
        OU_flat = params[2*n:]  # Extract OU (flattened)
        OU = OU_flat.reshape(n, n)  # Reshape into matrix

        Lambda = np.diag(lambda_diag)  # Convert to diagonal matrix

        log_likelihood = 0
        for t in range(1, T):
            mean = X[t-1] + Lambda @ (M - X[t-1]) * dt  # Mean transition
            cov = OU @ OU.T * dt  # Covariance matrix

            # Compute Gaussian log-likelihood
            log_likelihood += multivariate_normal.logpdf(X[t], mean=mean, cov=cov)

        return -log_likelihood  # We minimize the negative log-likelihood

    # **Initial Guesses**
    init_M = np.ones(n)*30  # Initial guess for M
    init_Lambda = np.ones(n)*0.04  # Initial guess for diagonal elements of Λ
    init_OU = np.eye(n).flatten()  # Initial guess for OU as identity matrix

    # Optimize using SciPy's minimization function
    init_params = np.concatenate([init_M, init_Lambda, init_OU])
    bounds = [(27, 33)]*n + [(0, 1)]*n + [(None, None)]*(n*n)  # No constraints
    result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)

    # Extract estimated parameters
    estimated_M = result.x[:n]
    estimated_Lambda = np.array(result.x[n:2*n]) * 252 # Convert diagonal elements to matrix
    estimated_OU = result.x[2*n:].reshape(n, n) * np.sqrt(252)

    coeffs['mu'] = estimated_M.tolist()
    coeffs['lambda'] = estimated_Lambda.tolist()
    coeffs['SigmaOU'] = estimated_OU.tolist()
    coeffs['x0'] = X[-1].tolist()
    return coeffs