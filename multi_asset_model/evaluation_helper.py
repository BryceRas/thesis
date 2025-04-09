import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def graph(data, ticker_list, model_version, run_number, beta, mu):
    num_lines = len(ticker_list)
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / num_lines) for i in range(num_lines)]

    data['date'] = pd.to_datetime(data['date'])

    fig, ax = plt.subplots(5,1, figsize=(15,15))
    ax[0].plot(data['date'],data['wealth_trained'], label='Polynomial Model', color='g')
    ax[0].plot(data['date'],data['wealth_base'], label='Base Model', color='b')
    ax[0].plot(data['date'],data['wealth'], label='Linear Model', color='r')
    # ax[0].set_title('Portfolio Wealth')
    ax[0].set_ylabel('Wealth', fontsize=12)
    ax[0].legend(loc='upper left')

    ax[1].set_ylabel('Risky Asset Price', fontsize=12)
    # ax[1].set_title('Asset Price')
    for i in range(len(ticker_list)):
        ax[1].plot(data['date'],data[f'{ticker_list[i]}_price'], label=f'{ticker_list[i]}', color=colors[i])
        ax[1].axhline(y =mu[i], color = colors[i], linestyle = '--', label = 'Historical Mean')
    ax[1].legend(loc='upper left')

    
    ax[2].axhline(y=0, color='black', linestyle='--', label='No Position')
    ax[2].set_title('Polynomial Model')
    ax[2].set_ylabel('Portion of Wealth in Risky Asset', fontsize=12)
    for i in range(len(ticker_list)):
        ax[2].plot(data['date'],data['portfolio_balance_trained'].apply(lambda x: x[i]), label=f'{ticker_list[i]}', color=colors[i])
    ax[2].legend(loc='upper left')

    ax[3].axhline(y=0, color='black', linestyle='--', label='No Position')
    ax[3].set_title('Base Model')
    ax[3].set_ylabel('Portion of Wealth in Risky Asset', fontsize=12)
    for i in range(len(ticker_list)):
        ax[3].plot(data['date'],data['portfolio_balance_base'].apply(lambda x: x[i]), label=f'{ticker_list[i]}', color=colors[i])
    ax[3].legend(loc='upper left')

    ax[4].axhline(y=0, color='black', linestyle='--', label='No Position')
    ax[4].set_title('Linear Model')
    ax[4].set_ylabel('Portion of Wealth in Risky Asset', fontsize=12)
    for i in range(len(ticker_list)):
        ax[4].plot(data['date'],data['portfolio_balance'].apply(lambda x: x[i]), label=f'{ticker_list[i]}', color=colors[i])
    ax[4].legend(loc='upper left')

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

    ax[0].set_title(f'Single Asset Model Evaluation for {ticker_list} with alpha = {beta}')

    plt.tight_layout()
    plt.savefig(f"evaluation_graphs/{ticker_list}_evaluation_model_{model_version}_run_{run_number}.png")