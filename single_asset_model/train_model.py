import argparse
import json
import torch
import os

from find_coefficients import collect_values
from pytorch_model import train

folder_path = "single_asset_model/training_graphs"
beta_default = 1e-4
end_year = 2020

def trainer(stock_ticker, run_number, beta):
    print(f"Processing : {stock_ticker}")
    file_path = f"single_asset_model/model_asset_coefficients/{stock_ticker}_coeffs_endyear_{end_year}.json"
    if os.path.exists(file_path):
        coeffs = json.load(open(file_path))
    else: coeffs = collect_values([stock_ticker], end_year)[stock_ticker]
    json.dump(coeffs, open(file_path, "w"))
    
    upper_model, lower_model = train(coeffs['lambda'], coeffs['mu'], 
                                     coeffs['sigma'], coeffs['x0'],run_number, beta)
    torch.save(upper_model.state_dict(), f"single_asset_model/models/{stock_ticker}_upper_model_{run_number}.pth")
    torch.save(lower_model.state_dict(), f"single_asset_model/models/{stock_ticker}_lower_model_{run_number}.pth")

def main():
    parser = argparse.ArgumentParser(description="Request a stock ticker and model version from the command line.")
    parser.add_argument("stock_ticker", type=str, help="The name of the stock ticker to process")
    parser.add_argument("model_version", type=str, help="The name of the stock ticker to process")
    args = parser.parse_args()
    stock_ticker = args.stock_ticker
    run_number = args.model_version

    trainer(stock_ticker, run_number, beta_default)
    

if __name__ == "__main__":
    main()
