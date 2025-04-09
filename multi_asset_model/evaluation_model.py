import argparse
import torch
import json
import pandas as pd
import numpy as np

from pytorch_helper import BoundaryNetwork
from evaluation_backtesting import backtest
from evaluation_helper import graph

folder_path = "evaluation_graphs"
hidden_dim = 64
beta_default = 1e-4
tickers = ['ADI', 'AEP', 'HAL']

def evaluate(model_name, run_number, beta, note, ticker_list=None):
    print(f"Evaluating model: {model_name} for {ticker_list}, run number: {run_number}")
    input_dim = len(ticker_list)*2 +1
    upper_model = BoundaryNetwork(input_dim, hidden_dim, output_dim=len(ticker_list)).to('cuda')
    upper_model.load_state_dict(torch.load(f"models/V{model_name}_upper_model_{note}.pth", weights_only=True))
    upper_model.eval()

    lower_model = BoundaryNetwork(input_dim, hidden_dim, output_dim=len(ticker_list)).to('cuda')
    lower_model.load_state_dict(torch.load(f"models/V{model_name}_lower_model_{note}.pth", weights_only=True))
    lower_model.eval()
    
    coeffs = json.load(open(f"model_coeffs/coeffs_{model_name}{note}.json"))

    trained_model_results = backtest(ticker_list, upper_model, lower_model, coeffs, beta, use_nn=True)
    print(f"Final wealth of trained model: {trained_model_results['wealth'].iloc[-1]}")
    base_model_results = backtest(ticker_list, upper_model, lower_model, coeffs, beta, use_nn=False)
    print(f"Final wealthof base model: {base_model_results['wealth'].iloc[-1]}")

    upper_model = BoundaryNetwork(input_dim, hidden_dim, output_dim=len(ticker_list)).to('cuda')
    upper_model.load_state_dict(torch.load(f"models/V0_upper_model_linear.pth", weights_only=True))
    upper_model.eval()

    lower_model = BoundaryNetwork(input_dim, hidden_dim, output_dim=len(ticker_list)).to('cuda')
    lower_model.load_state_dict(torch.load(f"models/V0_lower_model_linear.pth", weights_only=True))
    lower_model.eval()
    linear_model_results = backtest(ticker_list, upper_model, lower_model, coeffs, beta, use_nn=True)

    combined = pd.merge(trained_model_results, base_model_results, 
        on=[col for col in trained_model_results.columns if col not in 
            ['wealth', 'portfolio_balance', 'trading_costs_paid', 'carry_costs']], 
        suffixes=('_trained', '_base')
    )
    combined = combined.merge(linear_model_results,on=[col for col in linear_model_results.columns if col not in 
            ['wealth', 'portfolio_balance', 'trading_costs_paid', 'carry_costs']],)
    combined.to_csv(f"evaluation_data/evaluation_model_{model_name}_run_{run_number}.csv", index=False)
    graph(combined, ticker_list, model_name, run_number, beta, np.array(coeffs['mu']))
    

def main():
    parser = argparse.ArgumentParser(description="Request model number from the command line.")
    # parser = argparse.ArgumentParser(description="Request a model number from the command line.")
    parser.add_argument("model_number", type=str, help="The model number to evaluate.")
    parser.add_argument("run_number", type=str, help="The run number for model and ticker")
    parser.add_argument("note", type=str, help="model note")
    args = parser.parse_args()
    model_version, stock_ticker, run_number, note = args.model_number, tickers, args.run_number, args.note
    evaluate(model_version,  run_number, beta_default, note,stock_ticker)
    

if __name__ == "__main__":
    main()