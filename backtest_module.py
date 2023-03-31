# Import necessary libraries
import pandas as pd
import mplfinance as mpf
import pandas as pd
import numpy as np
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.constraints import MaxNorm
# from keras.optimizers import SGD
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense 
from keras.models import load_model
from datetime import datetime
import os
import glob
def find_matching_files(path, pattern):
    pattern = os.path.join(path, pattern)
    matching_files = []
    
    for file_path in glob.iglob(pattern, recursive=True):
        if os.path.isfile(file_path):
            matching_files.append(file_path)
    
    return matching_files
PDEBUG=False
def pdebug(err):
    if PDEBUG:
        print("\033[0;31m Debug msg: "+str(sys._getframe().f_code.co_name) +" \033[0;33m"+str(err))

MetaData=pd.DataFrame()
PRECISION = 0.0


BUY_PCT=0.5
SELL_PCT=0.3
PERIODE_START="2022-04-21 00:00:00"
PERIODE_END="2022-05-11 00:00:00"
start_period = pd.Timestamp(PERIODE_START)
end_period = pd.Timestamp(PERIODE_END)


#trading_options:
TAKE_PROFIT= 0.4  
STOP_LOSS=  0.8 
MAX_HOLDING_TIME= 16

USE_TRAILING_STOP_LOSS: False
TRAILING_STOP_LOSS= .002     
TRAILING_TAKE_PROFIT= .002  

PAIR_WITH= "USDT"
TRADE_TOTAL= 100 
TRADE_SLOTS= 5

TRADING_FEE= 0.1
  

def plot_ohlcv(df,title, start_date, end_date):
    """
    Plots OHLCV data using mplfinance library.

    Args:
    df (pd.DataFrame): Dataframe containing OHLCV data with columns ['date', 'open', 'high', 'low', 'close', 'volume'].
    start_date (str): Start date of the plot in the format 'YYYY-MM-DD'.
    end_date (str): End date of the plot in the format 'YYYY-MM-DD'.
    """

    # # Ensure the index is of type datetime and sorted
    # df['date'] = pd.to_datetime(df['date'])
    # df = df.set_index('date')
    # df = df.sort_index()

    # Filter the data between start_date and end_date
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df.loc[mask]

    # Set the plot size to full width
    fig_width = 25
    fig_height = 9

    # Plot the OHLCV chart
    mpf.plot(filtered_df, type='candle', style='charles', volume=True, title=title, ylabel='Price', ylabel_lower='Volume', figratio=(fig_width, fig_height),  tight_layout=True)

def find_worst_period(df, window):
    """
    Find the worst period with the largest price drop in the given OHLCV DataFrame.

    Args:
    df (pd.DataFrame): Dataframe containing OHLCV data with columns ['date', 'open', 'high', 'low', 'close', 'volume'].
    window (int): Window size for calculating the largest price drop.

    Returns:
    (pd.Timestamp, pd.Timestamp): Start and end date of the worst period.
    """


    # Calculate rolling minimum close prices
    min_close = df['close'].rolling(window=window).min()

    # Find the largest price drop within the specified window
    largest_drop = (df['close'] / min_close.shift(1) - 1).idxmin()

    # Calculate start and end dates of the worst period
    start_date = largest_drop - pd.Timedelta(days=window-1)
    end_date = largest_drop

    return start_date, end_date



def buy_alwase(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    df['buy']=1
    return df

def mini_expand5(pair="GMT/USDT", i=0, j=10000, window=2, metadata=MetaData,
                 high_weight=1, BUY_PCT=0.5, SELL_PCT=0.3,
                 buy_function=buy_alwase):
    start_index=i
    end_index=j
    window_size=window
    buy_fn=buy_function
    """
    This function takes in several parameters to calculate technical indicators and returns a merged dataframe.
    
    :param pair: str, default "GMT/USDT"
        The trading pair to analyze.
        
    :param start_index: int, default 0
        The start index for selecting data.
        
    :param end_index: int, default 10000
        The end index for selecting data.
    
    :param window_size: int, default 2
        The window size to use for analyzing the data.
    
    :param metadata: MetaData
        The metadata to use for analyzing the data.
    
    :param high_weight: int, default 1
        The weight to use for calculating the high.
    
    :param BUY_PCT: float, default BUY_PCT
        The buy pct to use for analyzing the data.
    
    :param SELL_PCT: float, default SELL_PCT
        The sell pct to use for analyzing the data.
    
    :param buy_fn: function, default buy_min_up
        The buy function to use for analyzing the data.
    
    :return: pd.DataFrame
        A merged dataframe containing the calculated technical indicators.
    """
    pdebug(f"mini_expand : {pair}")
    # Select data
    pair_df = df_list1m[pair].iloc[start_index:end_index]
    btc_df = df_list1m["BTC/USDT"].loc[(pair_df.index[0] - pd.DateOffset(days=window_size+1)).round(freq='1 min'):pair_df.index[-1]+pd.Timedelta(f"{window_size} day")]
    # Calculate technical indicators
    pair_full = full_expand(pair_df, df_list5m[pair], df_list15m[pair], df_list1h[pair], df_list1d[pair], window_size)
    btc_full = full_expand(btc_df, df_list5m["BTC/USDT"], df_list15m["BTC/USDT"], df_list1h["BTC/USDT"], df_list1d["BTC/USDT"], window_size)   
    btc_full = btc_full.add_prefix("BTC_")
    merged = pd.merge(pair_full, btc_full, left_index=True, right_index=True)
    day_expand(merged)
    Meta_expand(merged, metadata, pair)
    merged = buy_fn(merged, BUY_PCT=BUY_PCT, SELL_PCT=SELL_PCT, window=MAX_FORCAST_SIZE)
    merged["high"] = (merged["open"] + high_weight * merged["high"] + merged["low"] + merged["close"]) / (3 + high_weight)
    merged["BTC_high"] = (merged["BTC_open"] + high_weight * merged["BTC_high"] + merged["BTC_low"] + merged["BTC_close"]) / (3 + high_weight)
    merged.rename(columns={"high":"price"},inplace=True)
    merged.rename(columns={"BTC_high":"BTC_price"},inplace=True)
    merged = merged.drop(columns=["BTC_open","BTC_low","BTC_close","open","low","close"])
    open_high_low_close_cols = merged.columns.str.contains("open|high|low|close")
    # merged.loc[:, open_high_low_close_cols & merged.columns.str.contains("BTC")] = (
    #     (merged["BTC_price"] - merged.loc[:, open_high_low_close_cols & merged.columns.str.contains("BTC")]) / merged["BTC_price"]
    # )
    # merged.loc[:, open_high_low_close_cols & ~merged.columns.str.contains("BTC")] = (
    #     (merged["price"] - merged.loc[:, open_high_low_close_cols & ~merged.columns.str.contains("BTC")]) / merged["price"]
    # )
    for key in merged.keys():
        if key.find("BTC")!=-1 and (key.find("open")!=-1 or
    key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            merged[key]=(merged["BTC_price"]-merged[key])/merged["BTC_price"]
        if key.find("BTC")==-1 and (key.find("open")!=-1 or
    key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            merged[key]=(merged["price"]-merged[key])/merged["price"]

    merged=merged.dropna()
    pdebug(f'######################  mini_expand5 {pair} - shape {merged.shape}  buy mean : {hp(merged.buy.mean())} ############################')
    return merged


def create_portfolio_dataframe(num_slots, init_quantity=500, base_currency='usdt', PERIODE_START=PERIODE_START):
    # create an empty DataFrame with the columns we need
    columns = ['date', f'reserve_{base_currency}', f'total_{base_currency}']
    for i in range(num_slots):
        columns += [f'slot{i+1}_symbol', f'slot{i+1}_volume', f'slot{i+1}_original_price_{base_currency}', f'slot{i+1}_current_total_{base_currency}', f'slot{i+1}_bought_time']
    df = pd.DataFrame(columns=columns)
    # create a sample data
    for date in pd.date_range(PERIODE_START, periods=1):
        data = {'date': date, f'total_{base_currency}': init_quantity}
        for i in range(num_slots):
            data[f'slot{i+1}_bought_time'] = None
        df = df.append(data, ignore_index=True)
    # set the date column as the index of the DataFrame
    df.set_index('date', inplace=True)
    df.iloc[:, 0] = np.float64(init_quantity)
    return df.iloc[0:1]




def generate_signals(ALLTOP20VOLUMES, df_list1m, WINDOW_SIZE, MetaData, TAKE_PROFIT, STOP_LOSS, backtest_model):
    SIGNAL_DF = pd.DataFrame(columns=['coin', 'time', 'price', 'note'])
    for day, TOPLIST in ALLTOP20VOLUMES.items():
        for coin in TOPLIST:
            try:
                pdebug(f">>>>>>>>>>> working on {coin} at: {day} :")
                loc_start = df_list1m[coin].index.get_loc(day)
                loc_end = df_list1m[coin].index.get_loc(day+pd.Timedelta('1 day'))
                gc.collect()
                df = mini_expand5(pair=coin, i=loc_start, j=loc_end, window=WINDOW_SIZE, metadata=MetaData, BUY_PCT=TAKE_PROFIT, SELL_PCT=STOP_LOSS, buy_function=buy_alwase)
                dt = df.iloc[:,:-1].to_numpy(dtype=np.float32)
                predictions_note = backtest_model.predict(dt)
                predictions_round = predictions_note.round()
                dico_signal = {"coin":coin, "time":df[predictions_round==1].index.values, "price":df[predictions_round==1]["price"].values, "note":predictions_note[predictions_round==1]}
                df_signal_coin = pd.DataFrame(dico_signal)
                SIGNAL_DF = pd.concat([SIGNAL_DF, df_signal_coin])
            except:
                pdebug(f"error at {day} in {coin}")
    return SIGNAL_DF

import gc
import pandas as pd

def generate_signals_optimized(ALLTOP20VOLUMES, df_list1m, WINDOW_SIZE, MetaData, TAKE_PROFIT, STOP_LOSS, backtest_model):
    SIGNAL_DF = pd.DataFrame(columns=['coin', 'time', 'price', 'note'])
    
    # Find all unique pairs in ALLTOP20VOLUMES
    unique_pairs = set(coin for top_list in ALLTOP20VOLUMES.values() for coin in top_list)

    # Find the min and max timestamp in ALLTOP20VOLUMES
    min_timestamp = min(ALLTOP20VOLUMES)
    max_timestamp = max(ALLTOP20VOLUMES)

    # Calculate mini_expand5 and generate signals for all unique pairs in the maximum time range
    for coin in unique_pairs:
        print(f">>>>>>>>>>> working on {coin} :")
        try:
            # Compare the min and max timestamp with the index of df_list1m
            start_date = max(min_timestamp, df_list1m[coin].index[0])
            end_date = min(max_timestamp, df_list1m[coin].index[-1])

            loc_start = df_list1m[coin].index.get_loc(start_date, method='nearest')
            loc_end = df_list1m[coin].index.get_loc(end_date, method='nearest')
            
            gc.collect()
            df = mini_expand5(pair=coin, i=loc_start, j=loc_end, window=WINDOW_SIZE, metadata=MetaData, BUY_PCT=TAKE_PROFIT, SELL_PCT=STOP_LOSS, buy_function=buy_alwase)
            
            dt = df.iloc[:, :-1].to_numpy(dtype=np.float32)
            
            # Preprocessing: Handle NaN and Inf values
            dt = np.nan_to_num(dt, nan=0.0)
            dt = np.clip(dt, -1e12, 1e12)
            predictions_note = backtest_model.predict(dt)
            predictions_round = predictions_note.round()
            
            for day in ALLTOP20VOLUMES:
                        start_of_day = day.replace(hour=0, minute=0)
                        end_of_day = day.replace(hour=23, minute=59)

                        for minute in pd.date_range(start_of_day, end_of_day, freq='1T'):
                            if minute in ALLTOP20VOLUMES and coin in ALLTOP20VOLUMES[minute]:
                                signal = predictions_round[df.index.get_loc(minute)]
                                note = predictions_note[df.index.get_loc(minute)]

                                if signal == 1:
                                    SIGNAL_DF = SIGNAL_DF.append({"coin": coin, "time": minute, "price": df.loc[minute, "price"], "note": note}, ignore_index=True)


        except Exception as e:
            print(f"Error while proceding {coin} : {e}")

    return SIGNAL_DF


def get_top_volumes(start_period,end_period):
    ALLTOP20VOLUMES={}
    for day in df_list1d["BTC/USDT"].index:
        if start_period<= day <= end_period:
            Top20vol={}
            for p,df in df_list1d.items():
                if p not in ["EUR/USDT","EUR/BUSD","BTC/USDT","BTC/BUSD"]:
                    try:Top20vol.update({p:(df.loc[day].volume*df.loc[day].close)})
                    except Exception as e :pdebug(f'Time Error wile working on {p}: {e}')
                    #print (Top20vol)
            ALLTOP20VOLUMES.update({day:sorted(Top20vol, key=Top20vol.get, reverse=True)[:20]})
    return ALLTOP20VOLUMES

def is_coin_in_portfolio(symbol,instant,PORTFOLIO):
    for i in range(1,TRADE_SLOTS+1):
        try:
            if (PORTFOLIO[f"slot{i}_symbol"].loc[instant]==symbol):
                return i
        except Exception as e:
            print(f"error :{e}")
    return False


ORDER_HISTORY = pd.DataFrame(columns=['Order ID', 'Pair', 'Side', 'Price', 'Quantity', 'Executed', 'Time', 'Status'])
order_id_counter = 0

def add_order_to_history(pair, side, price, quantity, executed, time, status):
    global order_id_counter
    global ORDER_HISTORY

    order_id_counter += 1
    order = {'Order ID': order_id_counter,
             'Pair': pair,
             'Side': side,
             'Price': price,
             'Quantity': quantity,
             'Executed': executed,
             'Time': time,
             'Status': status}
    ORDER_HISTORY = ORDER_HISTORY.append(order, ignore_index=True)
    
    
    

def first_empty_slot(time, PORTFOLIO):
    for i in range(1, TRADE_SLOTS + 1):
        if pd.isna(PORTFOLIO.at[time, f'slot{i}_symbol']):
            return i
    return False



def buy_coin(time, coin, price, PORTFOLIO, pair_with_qte=TRADE_TOTAL):
    # pdebug("buy coin")
    slot_num = is_coin_in_portfolio(coin, time, PORTFOLIO)
    current_reserve = PORTFOLIO.at[time, f'reserve_{PAIR_WITH}']
    slot_ft = first_empty_slot(time, PORTFOLIO)
    
    if (not slot_num) and (current_reserve >= pair_with_qte):
        PORTFOLIO.at[time, f'reserve_{PAIR_WITH}'] = current_reserve - pair_with_qte
        PORTFOLIO.at[time, f'total_{PAIR_WITH}'] = PORTFOLIO.at[time, f'total_{PAIR_WITH}'] - TRADING_FEE * pair_with_qte / 100
        PORTFOLIO.at[time, f'slot{slot_ft}_symbol'] = coin
        PORTFOLIO.at[time, f'slot{slot_ft}_volume'] = (pair_with_qte - (TRADING_FEE * pair_with_qte / 100)) / price
        PORTFOLIO.at[time, f'slot{slot_ft}_original_price_{PAIR_WITH}'] = price
        PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}'] = price * PORTFOLIO.at[time, f'slot{slot_ft}_volume']
        PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = time  # Add bought time for the slot
        add_order_to_history(pair=f"{coin}_{PAIR_WITH}", side='buy', price=price, quantity=pair_with_qte, executed=pair_with_qte, time=time, status='filled')
        pdebug(f"Good condition for buying {PORTFOLIO.at[time, f'slot{slot_ft}_volume']} x {coin} at {price}")
        # pdebug(PORTFOLIO.tail)
    else:
        pdebug(f"xxx We cannot buy {coin} at {price} in {time} xxx")
    return PORTFOLIO


def update_slots(time, df_list1m, PORTFOLIO):
    # pdebug("update_slots")
    if time - pd.Timedelta("1 minute") in PORTFOLIO.index:
        previous_minute = time - pd.Timedelta("1 minute")
    else:
        return PORTFOLIO  # Exit the function if previous_minute is not in the DataFrame index

    previous_reserve = PORTFOLIO.at[previous_minute, f'reserve_{PAIR_WITH}']
    previous_total = PORTFOLIO.at[previous_minute, f'total_{PAIR_WITH}']

    PORTFOLIO.at[time, f'reserve_{PAIR_WITH}'] = previous_reserve
    PORTFOLIO.at[time, f'total_{PAIR_WITH}'] = previous_total

    for slot_ft in range(1, TRADE_SLOTS + 1):
        if not pd.isna(PORTFOLIO.at[previous_minute, f'slot{slot_ft}_symbol']):
            # Get previous data
            previousslot=PORTFOLIO.at[previous_minute, f'slot{slot_ft}_symbol']
            # print(f"update_slots ->  previous slot {previousslot}")
            coin = PORTFOLIO.at[previous_minute, f'slot{slot_ft}_symbol']
            bought_at = PORTFOLIO.at[previous_minute, f'slot{slot_ft}_original_price_{PAIR_WITH}']
            previous_total_slot_value = PORTFOLIO.at[previous_minute, f'slot{slot_ft}_current_total_{PAIR_WITH}']
            slot_volume = PORTFOLIO.at[previous_minute, f'slot{slot_ft}_volume']
            # Computation 
            new_total_slot_value = df_list1m[coin].at[time, "close"] * slot_volume
            slot_value_difference = new_total_slot_value - previous_total_slot_value
            new_total_assets_value = previous_total + slot_value_difference
            pdebug(f"update_slots -> new_total_assets_value : {new_total_assets_value}")
            # Update the slot using the new price from df_list1m dataset
            PORTFOLIO.at[time, f'slot{slot_ft}_symbol'] = coin
            PORTFOLIO.at[time, f'slot{slot_ft}_original_price_{PAIR_WITH}'] = bought_at
            PORTFOLIO.at[time, f'slot{slot_ft}_volume'] = slot_volume
            PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = PORTFOLIO.at[previous_minute, f'slot{slot_ft}_bought_time']
            PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}'] = new_total_slot_value
            PORTFOLIO.at[time, f'total_{PAIR_WITH}'] = new_total_assets_value  
            
        else:
            # print(f"no coin in slot {slot_ft}")
            pass         
    return PORTFOLIO

def stop_loss_or_take_profit(time, df_list1m, PORTFOLIO, st_pct, tp_pct, trade_fee_ptc):
    pdebug(f"stop_loss at {st_pct }% or take_profit at {tp_pct}% with trade_fee_ptc at {trade_fee_ptc}%")
    for slot_ft in range(1, TRADE_SLOTS+1):
        if pd.isna(PORTFOLIO.at[time, f"slot{slot_ft}_symbol"]):
            pdebug(f"no coin in slot {slot_ft}")
            continue
        coin = PORTFOLIO.at[time, f"slot{slot_ft}_symbol"]
        # Check if stop loss or take profit order should be triggered
        bought_price = PORTFOLIO.at[time, f"slot{slot_ft}_original_price_{PAIR_WITH}"]
        slot_volume = PORTFOLIO.at[time, f"slot{slot_ft}_volume"]
        current_price = df_list1m[coin].at[time, "close"]
        slot_total_value = PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}']  #current_price * slot_volume
        slot_fee=(trade_fee_ptc/100 * slot_total_value)
        slot_current_profit = ((current_price - bought_price) * slot_volume)# - slot_fee
        slot_current_profit_pct = (slot_current_profit / (bought_price * slot_volume)) * 100
        if slot_current_profit_pct < -st_pct:
            # Trigger stop loss
            pdebug(f"coin: {coin} in slot {slot_ft}")

            PORTFOLIO.at[time, f"slot{slot_ft}_symbol"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_volume"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_original_price_{PAIR_WITH}"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_current_total_{PAIR_WITH}"] = np.nan
            PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = None
            PORTFOLIO.at[time, f"total_{PAIR_WITH}"] -=   slot_fee
            PORTFOLIO.at[time, f"reserve_{PAIR_WITH}"] += (slot_total_value - slot_fee)
            add_order_to_history(pair=f"{coin}_{PAIR_WITH}", side='sell', price=current_price, quantity=slot_volume, executed=slot_volume, time=time, status='filled')
            pdebug(f"Stop loss triggered for {coin} in slot {slot_ft}")

        elif slot_current_profit_pct > tp_pct:
            # Trigger take profit
            PORTFOLIO.at[time, f"slot{slot_ft}_symbol"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_volume"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_original_price_{PAIR_WITH}"] = np.nan
            PORTFOLIO.at[time, f"slot{slot_ft}_current_total_{PAIR_WITH}"] = np.nan
            PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = None
            PORTFOLIO.at[time, f"total_{PAIR_WITH}"] -=   slot_fee
            PORTFOLIO.at[time, f"reserve_{PAIR_WITH}"] += (slot_total_value - slot_fee)
            add_order_to_history(pair=f"{coin}_{PAIR_WITH}", side='sell', price=current_price, quantity=slot_volume, executed=slot_volume, time=time, status='filled')
            pdebug(f"Take profit triggered for {coin} in slot {slot_ft}")

    return PORTFOLIO


def stop_holding(time, PORTFOLIO, max_holding_duration):
    max_holding_duration_td = pd.Timedelta(minutes=max_holding_duration)

    for slot_ft in range(1, TRADE_SLOTS + 1):
        if not pd.isna(PORTFOLIO.at[time, f'slot{slot_ft}_symbol']):
            coin = PORTFOLIO.at[time, f'slot{slot_ft}_symbol']
            bought_time = PORTFOLIO.at[time, f'slot{slot_ft}_bought_time']
            holding_duration = time - bought_time

            if holding_duration > max_holding_duration_td:
                # Sell the coin
                current_price = df_list1m[coin].at[time, "close"]
                slot_volume = PORTFOLIO.at[time, f'slot{slot_ft}_volume']
                slot_total_value = PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}']
                slot_fee = (TRADING_FEE / 100) * slot_total_value
                
                PORTFOLIO.at[time, f'slot{slot_ft}_symbol'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_volume'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_original_price_{PAIR_WITH}'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = np.nan
                
                PORTFOLIO.at[time, f'total_{PAIR_WITH}'] -= slot_fee
                PORTFOLIO.at[time, f'reserve_{PAIR_WITH}'] += (slot_total_value - slot_fee)
                
                add_order_to_history(pair=f"{coin}_{PAIR_WITH}", side='sell', price=current_price, quantity=slot_volume, executed=slot_volume, time=time, status='filled')
                pdebug(f"Automatic sell triggered for {coin} in slot {slot_ft} due to exceeding holding duration")
    return PORTFOLIO


SORT_SIGNAL=True
def backtest_buy_sell(SIGNAL_DF, df_list1m, start_period, end_period):
    pdebug(f"---   Backtesting form {start_period} to {end_period} while tp:{TAKE_PROFIT}% and sl:{STOP_LOSS}% and trading fee: {TRADING_FEE}% --- ")
    # Initialize variables and counters
    global ORDER_HISTORY 
    ORDER_HISTORY = pd.DataFrame(columns=['Order ID', 'Pair', 'Side', 'Price', 'Quantity', 'Executed', 'Time', 'Status'])
    current_time = pd.Timestamp(start_period)
    end_period_margin = end_period + pd.Timedelta(f"{2 * MAX_HOLDING_TIME} minute")
    PORTFOLIO = create_portfolio_dataframe(TRADE_SLOTS, init_quantity=550, base_currency=PAIR_WITH, PERIODE_START=start_period)
    PeriodRange = pd.date_range(start=pd.Timestamp(start_period), end=pd.Timestamp(end_period), freq='1min')

    # Verify buying opportunities
    for current_time in PeriodRange:
        if current_time == pd.Timestamp(start_period):
            pdebug(f"update_slots -> The first case : {current_time}")
            PORTFOLIO.loc[current_time + pd.Timedelta("1 minute")] =PORTFOLIO.loc[current_time]
            continue
        else:
            previous_minute = current_time - pd.Timedelta("1 minute")
        # print(PORTFOLIO.tail)
        PORTFOLIO=update_slots(current_time, df_list1m, PORTFOLIO=PORTFOLIO)
        pdebug(current_time)
        if SORT_SIGNAL:
            minute_signal_list=SIGNAL_DF[SIGNAL_DF.time == pd.Timestamp(current_time)].sort_values('note',ascending=False)
        else:
            minute_signal_list=SIGNAL_DF[SIGNAL_DF.time == pd.Timestamp(current_time)]
        for i, sig_row in minute_signal_list.iterrows():
            # Buy the coin if slot and cash is available:
            coin = sig_row['coin']
            time = sig_row['time']
            price = sig_row['price']  # We assume there's only one price per signal
            note = sig_row['note']
            slot_num = is_coin_in_portfolio(coin, previous_minute, PORTFOLIO)

            # Buy first:
            if (note + PRECISION) >= 0.5:
                PORTFOLIO = buy_coin(time, coin, price, PORTFOLIO=PORTFOLIO, pair_with_qte=TRADE_TOTAL)

        if USE_TRAILING_STOP_LOSS:
            PORTFOLIO = trailing_stop_loss_take_profit(current_time, PORTFOLIO, TAKE_PROFIT, STOP_LOSS, TRAILING_STOP_LOSS, TRAILING_TAKE_PROFIT)
        else:
            PORTFOLIO = stop_loss_or_take_profit(current_time, df_list1m, PORTFOLIO, STOP_LOSS, TAKE_PROFIT, TRADING_FEE)
        PORTFOLIO = stop_holding(current_time, PORTFOLIO, MAX_HOLDING_TIME)
        pct_benefit = 100 * PORTFOLIO.at[current_time, f'total_{PAIR_WITH}'] / PORTFOLIO[f'total_{PAIR_WITH}'].iloc[0]
        pdebug(f"Total benefit pct {pct_benefit}")
        
    return PORTFOLIO










import matplotlib.pyplot as plt

def back_test_metrics_v1(PORTFOLIO):
    initial_investment = PORTFOLIO.at[PORTFOLIO.index[0], f"reserve_{PAIR_WITH}"]
    final_value = PORTFOLIO.at[PORTFOLIO.index[-1], f"total_{PAIR_WITH}"]
    profit = final_value - initial_investment
    roi = (profit / initial_investment) * 100

    print(f"Initial Investment: {initial_investment:.2f} {PAIR_WITH}")
    print(f"Final Value: {final_value:.2f} {PAIR_WITH}")
    print(f"Profit: {profit:.2f} {PAIR_WITH}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

    # Draw plot of total assets
    plt.plot(PORTFOLIO.index, PORTFOLIO[f"total_{PAIR_WITH}"])
    plt.xlabel("Time")
    plt.ylabel(f"Total {PAIR_WITH}")
    plt.title("Total Assets Value Over Time")
    plt.grid()
    plt.show()

# Example usage:
# back_test_metrics(PORTFOLIO)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def back_test_metrics(PORTFOLIO):
    initial_investment = PORTFOLIO.at[PORTFOLIO.index[0], f"reserve_{PAIR_WITH}"]
    final_value = PORTFOLIO.at[PORTFOLIO.index[-1], f"total_{PAIR_WITH}"]
    profit = final_value - initial_investment
    roi = (profit / initial_investment) * 100

    print(f"Initial Investment: {initial_investment:.2f} {PAIR_WITH}")
    print(f"Final Value: {final_value:.2f} {PAIR_WITH}")
    print(f"Profit: {profit:.2f} {PAIR_WITH}")
    print(f"Return on Investment (ROI): {roi:.2f}%")

    # Calculate the 10 biggest wins and losses
    PORTFOLIO['change'] = PORTFOLIO[f"total_{PAIR_WITH}"].diff()
    PORTFOLIO['change'] = pd.to_numeric(PORTFOLIO['change'], errors='coerce')
    top_10_wins = PORTFOLIO.nlargest(10, 'change')
    top_10_losses = PORTFOLIO.nsmallest(10, 'change')

    # Draw plot of total assets
    plt.plot(PORTFOLIO.index, PORTFOLIO[f"total_{PAIR_WITH}"], label="Total Assets")
    plt.scatter(top_10_wins.index, top_10_wins[f"total_{PAIR_WITH}"], color='g', label='Top 10 Wins')
    plt.scatter(top_10_losses.index, top_10_losses[f"total_{PAIR_WITH}"], color='r', label='Top 10 Losses')
    plt.xlabel("Time")
    plt.ylabel(f"Total {PAIR_WITH}")
    plt.title("Total Assets Value Over Time")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
# back_test_metrics(PORTFOLIO)
def calculate_wins_losses(trade_history):
    wins = 0
    losses = 0

    open_positions = {}

    for index, trade in trade_history.iterrows():
        symbol = trade['Pair'].split('_')[0]
        side = trade['Side']
        price = trade['Price']

        if side == 'buy':
            open_positions[symbol] = price
        elif side == 'sell' and symbol in open_positions:
            buy_price = open_positions[symbol]

            if price > buy_price:
                wins += 1
            else:
                losses += 1

            del open_positions[symbol]
    print(f"total trades {wins+losses} whith {wins} win ({(100*wins/(wins+losses)):.2f}%) and {losses} losses ({(100*losses/(wins+losses)):.2f}%) ")
    return wins, losses









def get_back_test_metrics(PORTFOLIO):
    initial_investment = PORTFOLIO.at[PORTFOLIO.index[0], f"reserve_{PAIR_WITH}"]
    final_value = PORTFOLIO.at[PORTFOLIO.index[-1], f"total_{PAIR_WITH}"]
    profit = final_value - initial_investment
    roi = (profit / initial_investment) * 100

    # Calculate the 10 biggest wins and losses
    PORTFOLIO['change'] = PORTFOLIO[f"total_{PAIR_WITH}"].diff()
    PORTFOLIO['change'] = pd.to_numeric(PORTFOLIO['change'], errors='coerce')
    top_10_wins = PORTFOLIO.nlargest(10, 'change')
    top_10_losses = PORTFOLIO.nsmallest(10, 'change')

    metrics = {
        'initial_investment': initial_investment,
        'final_value': final_value,
        'profit': profit,
        'roi': roi,
        'top_10_wins': top_10_wins,
        'top_10_losses': top_10_losses
    }

    return metrics

# # Call the function and get the metrics dictionary
# metrics = get_back_test_metrics(PORTFOLIO)

# # Print the metrics
# print(f"Initial Investment: {metrics['initial_investment']:.2f} {PAIR_WITH}")
# print(f"Final Value: {metrics['final_value']:.2f} {PAIR_WITH}")
# print(f"Profit: {metrics['profit']:.2f} {PAIR_WITH}")
# print(f"Return on Investment (ROI): {metrics['roi']:.2f}%")

# # Use the metrics dictionary in other parts of your code
# top_10_wins = metrics['top_10_wins']
# top_10_losses = metrics['top_10_losses']



def tuner(SIGNAL_DF, df_list1m, start_period, end_period, tp_range, sl_range, mht_range):
    best_roi = -np.inf
    best_params = None
    best_metrics = None

    for tp in np.arange(*tp_range):
        for sl in np.arange(*sl_range):
            for mht in np.arange(*mht_range):
                global TAKE_PROFIT, STOP_LOSS, MAX_HOLDING_TIME
                TAKE_PROFIT = tp
                STOP_LOSS = sl
                MAX_HOLDING_TIME = mht

                PORTFOLIO = backtest_buy_sell(SIGNAL_DF, df_list1m, start_period=start_period, end_period=end_period)
                metrics = get_back_test_metrics(PORTFOLIO)

                if metrics['roi'] > best_roi:
                    best_roi = metrics['roi']
                    best_params = (tp, sl, mht)
                    best_metrics = metrics
                print('----------------------------------')
                print(f'for parameters TAKE_PROFIT:{TAKE_PROFIT} STOP_LOSS:{STOP_LOSS} MAX_HOLDING_TIME:{MAX_HOLDING_TIME} -> the ROI: {metrics["roi"]:.2f} , the profit: {metrics["profit"]:.2f} ,the final {metrics["final_value"]:.2f}  ')
                print('----------------------------------')
    return best_params, best_metrics















def trailing_stop_loss_take_profit(time, PORTFOLIO, initial_tp, initial_sl, trailing_stop_percentage, trailing_take_profit_percentage):
    global TRAILING_DATA
    for slot_ft in range(1, TRADE_SLOTS + 1):
        if not pd.isna(PORTFOLIO.at[time, f'slot{slot_ft}_symbol']):
            coin = PORTFOLIO.at[time, f'slot{slot_ft}_symbol']
            current_price = df_list1m[coin].at[time, "close"]

            if coin not in TRAILING_DATA['symbol'].values:
                TRAILING_DATA = TRAILING_DATA.append({'symbol': coin,
                                                      'trailing_stop_loss': current_price * (1 - initial_sl / 100),
                                                      'trailing_take_profit': current_price * (1 + initial_tp / 100)}, ignore_index=True)
            else:
                row_index = TRAILING_DATA[TRAILING_DATA['symbol'] == coin].index[0]
                if current_price > TRAILING_DATA.at[row_index, 'trailing_take_profit']:
                    TRAILING_DATA.at[row_index, 'trailing_take_profit'] = current_price * (1 + trailing_take_profit_percentage / 100)
                    TRAILING_DATA.at[row_index, 'trailing_stop_loss'] = current_price * (1 - trailing_stop_percentage / 100)
                elif current_price < TRAILING_DATA.at[row_index, 'trailing_stop_loss']:
                    TRAILING_DATA.at[row_index, 'trailing_stop_loss'] = current_price * (1 - trailing_stop_percentage / 100)

            trailing_stop_loss = TRAILING_DATA[TRAILING_DATA['symbol'] == coin]['trailing_stop_loss'].values[0]
            trailing_take_profit = TRAILING_DATA[TRAILING_DATA['symbol'] == coin]['trailing_take_profit'].values[0]

            if current_price <= trailing_stop_loss or current_price >= trailing_take_profit:
                # Sell the coin
                slot_volume = PORTFOLIO.at[time, f'slot{slot_ft}_volume']
                slot_total_value = PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}']
                slot_fee = (TRADING_FEE / 100) * slot_total_value

                PORTFOLIO.at[time, f'slot{slot_ft}_symbol'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_volume'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_original_price_{PAIR_WITH}'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_current_total_{PAIR_WITH}'] = np.nan
                PORTFOLIO.at[time, f'slot{slot_ft}_bought_time'] = np.nan

                PORTFOLIO.at[time, f'total_{PAIR_WITH}'] -= slot_fee
                PORTFOLIO.at[time, f'reserve_{PAIR_WITH}'] += (slot_total_value - slot_fee)

                add_order_to_history(pair=f"{coin}_{PAIR_WITH}", side='sell', price=current_price, quantity=slot_volume, executed=slot_volume, time=time, status='filled')
                pdebug(f"Trailing stop loss or take profit triggered for {coin} in slot {slot_ft}")
    return PORTFOLIO



def get_coingekko_list(coingekko_info_file):
    try:
        df = pd.read_csv(coingekko_info_file)
        return df
    except FileNotFoundError:
        url = 'https://api.coingecko.com/api/v3/coins/list'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df.to_csv(coingekko_info_file, index=False)
        return df

def uniq_coins_lower(pair_list):
    list(set([symbol.split('/')[0].replace("BUSD","").replace("USDT", "").lower() for symbol in ALL_USDT_BUSD_PAIRS] ))
    
def uniq_coins_upper(pair_list):
    list(set([symbol.split('/')[0].replace("BUSD","").replace("USDT", "").upper() for symbol in ALL_USDT_BUSD_PAIRS] ))
    
def uniq_coins(pair_list):
    list(set([symbol.split('/')[0].replace("BUSD","").replace("USDT", "").lower() for symbol in ALL_USDT_BUSD_PAIRS] ))
    
    
    