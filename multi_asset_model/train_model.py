import argparse
import json
import torch
import os

from find_coefficients import collect_values
from pytorch_model import train
from evaluation_backtesting import backtest

folder_path = "multi_asset_model/training_graphs"
beta_default = 1e-4
end_year = 2020

# tickers = ['ADI', 'AEP', 'HAL','C','CL','CMCSA','CSCO','D','F','FITB','HON','INTC','IP','IVZ','JCI']
tickers = ['ADI', 'AEP', 'HAL']

# returns final wealth of the model trained on last 2 years of testing data
def test_model(upper_model, lower_model, alpha, coeffs, stock_ticker,linear = False):
    results = backtest(stock_ticker, upper_model, lower_model, coeffs, alpha, use_nn=True, test=True, linear=linear)
    return results['wealth'].iloc[-1]

def trainer(model_number, beta, note='', ticker_list=None, attempts =1):
    if ticker_list == None: ticker_list = tickers
    best_test_return = 0
    doLinear = False
    print(f"Processing : {ticker_list}")
    if note == 'linear': doLinear = True
    coeffs = collect_values(ticker_list, end_year)
    json.dump(coeffs, open(f"model_coeffs/coeffs_{model_number}{note}.json", "w"))
    for a in range(attempts):
        print(f'Training Model [{a+1}/{attempts}]') 
        upper, lower = train(coeffs, model_number,a+1, beta, linear=doLinear)
        upper.eval()
        lower.eval()
        wealth_run = test_model(upper, lower, beta, coeffs, ticker_list, doLinear)
        if wealth_run > best_test_return:
            upper_model, lower_model, best_test_return = upper, lower, wealth_run
        else:
            os.remove(f'training_graphs/training_plot_{ticker_list}_model_{model_number}_{a+1}.png')
        print(f'current model wealth = {wealth_run} best model wealth = {best_test_return}')
    print('final model wealth =' , test_model(upper_model, lower_model, beta, coeffs, ticker_list, doLinear))
    torch.save(upper_model.state_dict(), f"models/V{model_number}_upper_model_{note}.pth")
    torch.save(lower_model.state_dict(), f"models/V{model_number}_lower_model_{note}.pth")
    # torch.save(upper_model.state_dict(), f"models/V0_upper.pth")
    # torch.save(lower_model.state_dict(), f"models/V0_lower.pth")

def main():
    print('enters main')
    parser = argparse.ArgumentParser(description="Request a model version from the command line.")
    # parser.add_argument("stock_ticker", type=str, help="The name of the stock ticker to process")
    parser.add_argument("model_version", type=str, help="The name of the stock ticker to process")
    args = parser.parse_args()
    # stock_ticker = args.stock_ticker
    model_number = args.model_version

    trainer(model_number, beta_default,'test')
    

if __name__ == "__main__":
    main()
