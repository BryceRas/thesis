import argparse
import torch
import json
import pandas as pd

from pytorch_helper import BoundaryNetwork
from evaluation_backtesting import backtest
from evaluation_helper import graph

folder_path = "evaluation_graphs"
hidden_dim = 64
input_dim = 4
beta_default = 1e-4
end_year =2020

def evaluate(stock_ticker, model_version, run_number, beta,note=''):
    print(f"Evaluating model: {model_version} for {stock_ticker}, run number: {run_number}")

    upper_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    upper_model.load_state_dict(torch.load(f"models/{stock_ticker}_upper_model_{model_version}{note}.pth", weights_only=True))
    upper_model.eval()

    lower_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    lower_model.load_state_dict(torch.load(f"models/{stock_ticker}_lower_model_{model_version}{note}.pth", weights_only=True))
    lower_model.eval()

    coeffs = json.load(open(f"model_asset_coefficients/{stock_ticker}_coeffs_endyear_{end_year}.json"))

    trained_model_results = backtest(stock_ticker, upper_model, lower_model, coeffs, beta, use_nn=True)
    print(f"Final wealth of trained model: {trained_model_results['wealth'].iloc[-1]}")
    base_model_results = backtest(stock_ticker, upper_model, lower_model, coeffs, beta, use_nn=False)
    print(f"Final wealth of base model: {base_model_results['wealth'].iloc[-1]}")

    # temp to add the lineaer model
    upper_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    upper_model.load_state_dict(torch.load("models/LAZ_upper_model_0_linear.pth", weights_only=True))
    upper_model.eval()

    lower_model = BoundaryNetwork(input_dim, hidden_dim).to('cuda')
    lower_model.load_state_dict(torch.load("models/LAZ_lower_model_0_linear.pth", weights_only=True))
    lower_model.eval()
    lin_model_results = backtest(stock_ticker, upper_model, lower_model, coeffs, beta, use_nn=True)
    print(f"Final wealth of linear trained model: {lin_model_results['wealth'].iloc[-1]}")
    # end of temp
    
    combined = pd.merge(trained_model_results, 
                        base_model_results, on=['date','price'], suffixes=('_trained', '_base')).merge(
                        lin_model_results,on=['date','price']
                        )
    combined.to_csv(f"evaluation_data/{stock_ticker}_evaluation_model_{model_version}_run_{run_number}.csv", index=False)
    graph(combined, stock_ticker, model_version, run_number, beta,coeffs['mu'])
    

def main():
    parser = argparse.ArgumentParser(description="Request a stock ticker and model number from the command line.")
    parser.add_argument("stock_ticker", type=str, help="The name of the stock ticker to process")
    # parser = argparse.ArgumentParser(description="Request a model number from the command line.")
    parser.add_argument("model_number", type=str, help="The model number to evaluate.")
    parser.add_argument("run_number", type=str, help="The run number for model and ticker")
    args = parser.parse_args()
    model_version, stock_ticker, run_number = args.model_number, args.stock_ticker, args.run_number
    evaluate(stock_ticker, model_version, run_number, beta_default)
    

if __name__ == "__main__":
    main()