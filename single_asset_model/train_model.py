import argparse
import json
import torch
import os

from find_coefficients import collect_values
from pytorch_model import train
from evaluation_backtesting import backtest

folder_path = "training_graphs"
alpha_default = 1e-4
end_year = 2020

# returns final wealth of the model trained on last 2 years of testing data
def test_model(upper_model, lower_model, alpha, coeffs, stock_ticker,linear = False):
    results = backtest(stock_ticker, upper_model, lower_model, coeffs, alpha, use_nn=True, test=True, linear=linear)
    return results['wealth'].iloc[-1]

def trainer(stock_ticker, run_number, alpha,note='',attempts =1):
    best_test_return = 0
    doLinear = False
    print(f"Processing : {stock_ticker}")
    file_path = f"model_asset_coefficients/{stock_ticker}_coeffs_endyear_{end_year}.json"
    if os.path.exists(file_path):
        coeffs = json.load(open(file_path))
    else: coeffs = collect_values([stock_ticker], end_year)[stock_ticker]
    json.dump(coeffs, open(file_path, "w"))
    if note == '_linear': doLinear = True
    for a in range(attempts):
        print(f'Training Model [{a+1}/{attempts}]') 
        upper, lower = train(coeffs['lambda'], coeffs['mu'], 
                                         coeffs['sigma'], coeffs['x0'],stock_ticker,run_number, alpha,linear=doLinear)
        upper.eval()
        lower.eval()
        wealth_run = test_model(upper, lower, alpha, coeffs, stock_ticker, doLinear)
        if wealth_run > best_test_return:
            upper_model, lower_model, best_test_return = upper, lower, wealth_run
        print(f'Current Model Wealth = {wealth_run} Best Model Wealth = {best_test_return}')
    wealth_final = test_model(upper_model, lower_model, alpha, coeffs, stock_ticker, doLinear)
    print(f'final wealth of best model = {wealth_final}')
    torch.save(upper_model.state_dict(), f"models/{stock_ticker}_upper_model_{run_number}{note}.pth")
    torch.save(lower_model.state_dict(), f"models/{stock_ticker}_lower_model_{run_number}{note}.pth")

def main():
    parser = argparse.ArgumentParser(description="Request a stock ticker and model version from the command line.")
    parser.add_argument("stock_ticker", type=str, help="The name of the stock ticker to process")
    parser.add_argument("model_version", type=str, help="The name of the stock ticker to process")
    args = parser.parse_args()
    stock_ticker = args.stock_ticker
    run_number = args.model_version

    trainer(stock_ticker, run_number, alpha_default)
    

if __name__ == "__main__":
    main()
