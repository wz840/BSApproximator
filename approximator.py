import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import getopt
import sys


def approximate(spot_price, strike_price, time_2_maturity, r, q, target, step_sigma, start_sigma, option='call'):
    """
   Do approximation by stepping forward / backward with step_sigma so as to get the fair price calculated
   by B-S model as close as target price and return estimated sigma(volatility)
   :param spot_price: spot price of underlying asset
   :param strike_price: strike price of the option
   :param time_2_maturity: time to maturity
   :param r: risk-free rate
   :param q: dividend ratio
   :param target: target spot price of option
   :param step_sigma: step sigma for dynamic trial (in bps)
   :param start_sigma: sigma for starting the trial
   :param option: option type
   :return: estimated sigma that makes the target as close as fair price
   """
    # determine direction first
    fair_price = bs_option(spot_price, strike_price, time_2_maturity, r, q, start_sigma, option)
    print('initial fair price = ' + str(fair_price))
    sigma_to_test = start_sigma
    if fair_price > target:
        # step backward
        while fair_price > target:
            sigma_to_test -= (step_sigma / 10000)
            fair_price = bs_option(spot_price, strike_price, time_2_maturity, r, q, sigma_to_test, option)
        return sigma_to_test
    elif fair_price < target:
        # step forward
        while fair_price < target:
            sigma_to_test += (step_sigma / 10000)
            fair_price = bs_option(spot_price, strike_price, time_2_maturity, r, q, sigma_to_test, option)
        return sigma_to_test
    else:
        return start_sigma


def bs_option(spot_price, strike_price, time_2_maturity, r, q, sigma, option='call'):
    """
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    q: rate of continuous dividend
    sigma: standard deviation of price of underlying asset
    """
    if time_2_maturity == 0:
        d1 = d2 = 0.5
    else:
        d1 = (np.log(spot_price / strike_price) + (r - q + 0.5 * sigma ** 2) * time_2_maturity) / (sigma * np.sqrt(time_2_maturity))
        d2 = (np.log(spot_price / strike_price) + (r - q - 0.5 * sigma ** 2) * time_2_maturity) / (sigma * np.sqrt(time_2_maturity))

    if option == 'call':
        return spot_price * np.exp(-q * time_2_maturity) * norm.cdf(d1, 0.0, 1.0) \
               - strike_price * np.exp(-r * time_2_maturity) * norm.cdf(d2, 0.0, 1.0)
    elif option == 'put':
        return strike_price * np.exp(-r * time_2_maturity) * norm.cdf(-d2, 0.0, 1.0) \
               - spot_price * np.exp(-q * time_2_maturity) * norm.cdf(-d1, 0.0, 1.0)
    else:
        return None


def read_file(file):
    """
    Read file to DataFrame
    :param file: file name
    :return: DataFrame containing all data in file
    """
    return pd.read_csv(file, header=0, index_col=0,
                       parse_dates=['Trading_date'],
                       infer_datetime_format=True,
                       keep_date_col=True)


def do_statistics(df):
    """
    Prepare DataFrame with dummy row values
    :param df: Raw DataFrame
    """
    length = len(df.index)
    df['estimated_sigma'] = [None] * length
    df['spot_price_using_estimation'] = [None] * length
    df['gap'] = [None] * length


def calculate_approximation(option_type, step_sigma, ignore_abnormal=True):
    """
    Calculate approximation
    :param option_type: option type call or put
    :param step_sigma: step sigma to increase or decrease in bps
    :param ignore_abnormal: flag to ignore invalid rows
    :return: Calculated DataFrame
    """
    if option_type == 'call':
        input_file = 'DT_call.csv'
        output_file = 'DT_call_output.csv'
        target_price_col = 'C'
    else:
        input_file = 'DT_put.csv'
        output_file = 'DT_put_output.csv'
        target_price_col = 'P'
    df = read_file(input_file)
    do_statistics(df)
    for index, row in df.iterrows():
        strike_price = float(row['K'])
        spot_price = float(row['S'])
        target_price = float(row[target_price_col])
        time_to_maturity = float(row['T'])
        dividend_ratio = float(row['q'])
        risk_free_rate = float(row['r'])
        start_sigma = float(row['EWMA_sigma'])
        if start_sigma == 0 or time_to_maturity == 0:
            # df.loc[index, 'estimated_sigma'] = start_sigma
            # df.loc[index, 'spot_price_using_estimation'] = target_price
            df.drop(index, inplace=True)
            print('Dropping EWMA_sigma=0 or T=0, index=' + str(index) + ', EWMA_sigma=' + str(start_sigma) + ', T=' + str(time_to_maturity))
        else:
            calculated_sigma = approximate(spot_price, strike_price, time_to_maturity, risk_free_rate, dividend_ratio, target_price,
                                           step_sigma, start_sigma, option_type)
            df.loc[index, 'estimated_sigma'] = approximate(spot_price, strike_price, time_to_maturity, risk_free_rate, dividend_ratio, target_price,
                                                           step_sigma, start_sigma, option_type)
            estimated_price = bs_option(spot_price, strike_price, time_to_maturity, risk_free_rate, dividend_ratio, calculated_sigma, option_type)
            if calculated_sigma < 0 and ignore_abnormal:
                df.drop(index, inplace=True)
                print('Dropping invalid estimated_sigma/price, index=' + str(index) + ', estimated_sigma=' + str(calculated_sigma))
                continue
            df.loc[index, 'spot_price_using_estimation'] = estimated_price
            df.loc[index, 'gap'] = float('{:.4f}'.format(abs(estimated_price - target_price)))
    df.to_csv(output_file, index_label='Trading_date', header=True)
    return df


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Deprecated
    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return:
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


if __name__ == '__main__':
    """
    README
    command:
        python approximator.py -t <call|put> -s <step length> --plotonly
        -t mandatory, option type, only call/put supported
        -s mandotory, approximate step in bps, int32, e.g. 1,2,3...
        --plotonly optional, when passing, script will only plot the figures using output file    
    """
    option = 'call'
    step = 1
    plot_only = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:s:', ['type', 'step', 'plotonly'])
    except getopt.GetoptError:
        print('Illegal arguments')
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-t', '--optiontype'):
            if arg not in ('call', 'put'):
                print('Illegal value of option type, only [call, put] supported!')
                sys.exit(1)
            option = arg
        elif opt in ('-s', '--step'):
            step = int(arg, 32)
        elif opt in '--plotonly':
            plot_only = True
    if option == 'call':
        title = '50ETF Call Option Volatility'
        figure_name = 'call_volatility.png'
        file_to_read = 'DT_call_output.csv'
    else:
        title = '50ETF Put Option Volatility'
        figure_name = 'put_volatility.png'
        file_to_read = 'DT_put_output.csv'
    if plot_only:
        df = read_file(file_to_read)
        df.plot(title=title, y=['EWMA_sigma', 'estimated_sigma'])
        plt.show()
    else:
        df = calculate_approximation(option, step)
        if option == 'call':
            title = '50ETF Call Option Volatility'
            figure_name = 'call_volatility.png'
        else:
            title = '50ETF Put Option Volatility'
            figure_name = 'put_volatility.png'
        df.plot(title=title, y=['EWMA_sigma', 'estimated_sigma'])
        plt.show()
