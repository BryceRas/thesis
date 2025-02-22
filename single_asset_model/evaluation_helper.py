import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def graph(data, stock_ticker, model_version, run_number, beta):
    data['date'] = pd.to_datetime(data['date'])

    fig, ax = plt.subplots(3,1, figsize=(15,10))
    ax[0].plot(data['date'],data['wealth_trained'], label='Trained Model', color='g')
    ax[0].plot(data['date'],data['wealth_base'], label='Base Model', color='b')
    # ax[0].set_title('Portfolio Wealth')
    ax[0].set_ylabel('Wealth', fontsize=12)
    ax[0].legend(loc='upper left')

    ax[1].plot(data['date'],data['price'])
    # ax[1].set_title('Asset Price')
    ax[1].set_ylabel('Risky Asset Price', fontsize=12)
    ax[1].axhline(y =30.07, color = 'r', linestyle = '--', label = 'Historical Mean')
    ax[1].legend(loc='upper left')

    ax[2].plot(data['date'],data['portfolio_balance_trained'], label='Trained Model Position', color='teal')
    ax[2].plot(data['date'],data['portfolio_balance_base'], label='Base Model Position', color='m')
    ax[2].axhline(y=0, color='lime', linestyle='--', label='No Position')
    # ax[2].set_title('Portfolio Balance')
    ax[2].set_ylabel('Portion of Wealth in Risky Asset', fontsize=12)
    ax[2].legend(loc='upper left')

    # Add year label only on January
    def format_date(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%b\n%Y') if dt.month == 1 else dt.strftime('%b')

    # Format x-axis
    for a in ax:
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # Major ticks on 1st of each month
        a.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for every month
        a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Show month as "Jan, Feb, etc."
        a.xaxis.set_minor_formatter(mdates.DateFormatter(''))  # Hide minor ticks
        a.xaxis.set_major_formatter(plt.FuncFormatter(format_date))  # Custom date format

    ax[0].set_title('Single Asset Model Evaluation for ' + stock_ticker + ' with beta = ' + str(beta))

    plt.tight_layout()
    plt.savefig(f"single_asset_model/evaluation_graphs/{stock_ticker}_evaluation_model_{model_version}_run_{run_number}.png")