import argparse
import torch
import json
import pandas as pd

from pytorch_helper import BoundaryNetwork
from evaluation_backtesting import backtest
from evaluation_helper import graph

folder_path = "single_asset_model/evaluation_graphs"
hidden_dim = 196
input_dim = 4
beta_default = 1e-4

def evaluate(model_version, stock_ticker, run_number, beta):
    print(f"Evaluating model: {model_version} for {stock_ticker}, run number: {run_number}")

    upper_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    upper_model.load_state_dict(torch.load(f"single_asset_model/models/{stock_ticker}_upper_{model_version}.pth", weights_only=True))
    upper_model.eval()

    lower_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    lower_model.load_state_dict(torch.load(f"single_asset_model/models/{stock_ticker}_lower_{model_version}.pth", weights_only=True))
    lower_model.eval()

    coeffs = json.load(open(f"single_asset_model/model_asset_coefficients/{stock_ticker}_coeffs_{model_version}.json"))

    trained_model_results = backtest(stock_ticker, upper_model, lower_model, coeffs, beta, use_nn=True)
    print(f"Final wealth of trained model: {trained_model_results['wealth'].iloc[-1]}")
    base_model_results = backtest(stock_ticker, upper_model, lower_model, coeffs, beta, use_nn=False)
    print(f"Final wealthof base model: {base_model_results['wealth'].iloc[-1]}")
    combined = pd.merge(trained_model_results, 
                        base_model_results, on=['date','price','ETF','beta'], suffixes=('_trained', '_base'))
    combined.to_csv(f"single_asset_model/evaluation_data/{stock_ticker}_evaluation_model_{model_version}_run_{run_number}.csv", index=False)
    graph(combined, stock_ticker, model_version, run_number, beta)
    

def main():
    parser = argparse.ArgumentParser(description="Request a stock ticker and model number from the command line.")
    parser.add_argument("stock_ticker", type=str, help="The name of the stock ticker to process")
    # parser = argparse.ArgumentParser(description="Request a model number from the command line.")
    parser.add_argument("model_number", type=str, help="The model number to evaluate.")
    parser.add_argument("run_number", type=str, help="The run number for model and ticker")
    args = parser.parse_args()
    model_version, stock_ticker, run_number = args.model_number, args.stock_ticker, args.run_number
    evaluate(model_version, stock_ticker, run_number, beta_default)
    

if __name__ == "__main__":
    main()