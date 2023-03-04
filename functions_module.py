from xdata_config import *

#DATA_FILE=DATA_DIR+'w'+str(WINDOW_SIZE)+'_EXTData.csv'

#DATA_FILE=DATA_DIR+'w'+str(WINDOW_SIZE)+'_EXTData.csv'
import sys
sys.path.append('/UltimeTradingBot/Crypto_backtest_tools')
# from utilities.get_data import get_historical_from_db
# from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics
import pandas as pd
import ccxt
import time
import matplotlib.pyplot as plt
#import ta
import numpy as np
import gc
import random
from sklearn.utils import shuffle
import seaborn as sns
import tensorflow as tf

from utilities.backtesting import plot_wallet_vs_asset, get_metrics, get_n_columns, basic_multi_asset_backtest, plot_sharpe_evolution, plot_bar_by_month
#from utilities.custom_indicators import SuperTrend
pd.options.mode.chained_assignment = None  # default='warn'
import gc
gc.collect()    
import ccxt
import matplotlib.pyplot as plt
import json
import numpy as np
import random
import os


def get_historical_from_db(exchange, symbol, timeframe, path="database/"):
    symbol = symbol.replace('/','-')
    df = pd.read_csv(filepath_or_buffer=path+str(exchange.name)+"/"+timeframe+"/"+symbol+".csv")
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df

def get_historical_from_path(path):
    df = pd.read_csv(filepath_or_buffer=path)
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df

def get_backtest_historical(exchange, symbol, timeframe, path="database/"):
    symbol = symbol.replace('/','-')
    df = pd.read_csv(filepath_or_buffer=path+"DataBackTest"+"/"+timeframe+"/"+symbol+".csv")
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df



import warnings
warnings.filterwarnings('ignore')
PRERR=False
def prerr(err):
    if PRERR:
        print("\033[0;31m Error in "+str(sys._getframe().f_code.co_name) +" \033[0;33m"+str(err))

PDEBUG=True
def pdebug(err):
    if PDEBUG:
        print("\033[0;31m Error in "+str(sys._getframe().f_code.co_name) +" \033[0;33m"+str(err))
        
        
Binance_USDT_HALAL = [
    "SNM/BUSD",
    "BTC/USDT",
    "LUNA/USDT",
    "ETH/USDT",
    "GMT/USDT",
    "UST/USDT",
    "SOL/USDT",
    "APE/USDT",
    "XRP/USDT",
    "IDEX/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "ADA/USDT",
    "JASMY/USDT",
    "TRX/USDT",
    "NEAR/USDT",
    "AXS/USDT",
    "GAL/USDT",
    "GALA/USDT",
    "SHIB/USDT",
    "ZIL/USDT",
    "ENS/USDT",
    "DOGE/USDT",
    "LTC/USDT",
    "EUR/USDT",
    "MANA/USDT",
    "DAR/USDT",
    "WAVES/USDT",
    "LAZIO/USDT",
    "ALICE/USDT",
    "ROSE/USDT",
    "ZEC/USDT",
    "ALGO/USDT",
    "GRT/USDT",
    "PSG/USDT",
    "SLP/USDT",
    "EOS/USDT",
    "PORTO/USDT",
    "ICP/USDT",
    "EGLD/USDT",
    "XMR/USDT",
    "KDA/USDT",
    "ETC/USDT",
    "MBOX/USDT",
    "OGN/USDT",
    "AR/USDT",
    "GLMR/USDT",
    "LOKA/USDT",
    "XLM/USDT",
    "MTL/USDT",
    "SNX/USDT",
    "PYR/USDT",
    "DASH/USDT",
    "CITY/USDT",
    "ASTR/USDT",
    "IOTA/USDT",
    "ACM/USDT",
    "BAR/USDT",
    "JUV/USDT",
    "SYS/USDT",
    "RVN/USDT",
    "MBL/USDT",
    "REN/USDT",
    "JST/USDT",
    "OMG/USDT",
    "ATM/USDT",
    "XEC/USDT",
    "STORJ/USDT",
    "ZRX/USDT",
    "SRM/USDT",
    "ICX/USDT",
    "API3/USDT",
    "ONT/USDT",
    "SKL/USDT",
    "MULTI/USDT",
    "QTUM/USDT",
    "COCOS/USDT",
    "VOXEL/USDT",
    "HIVE/USDT",
    "KP3R/USDT",
    "ATA/USDT",
    "STMX/USDT",
    "ADX/USDT",
    "HIGH/USDT",
    "NULS/USDT",
    "MLN/USDT",
    "YGG/USDT",
    "SC/USDT",
    "CKB/USDT",
    "TOMO/USDT",
    "STX/USDT",
    "FLUX/USDT",
    "DNT/USDT",
    "ORN/USDT",
    "PLA/USDT",
    "BADGER/USDT",
    "DF/USDT",
    "MOB/USDT",
    "LPT/USDT",
    "SCRT/USDT",
    "RAD/USDT",
    "NMR/USDT",
    "ELF/USDT",
    "TORN/USDT",
    "T/USDT",
    "QUICK/USDT",
    "LSK/USDT",
    "FIDA/USDT",
    "XNO/USDT",
    "BTG/USDT",
    "GHST/USDT",
    "EPS/USDT"
]

pair_list = Binance_USDT_HALAL
#tf = '1m'
oldest_pair = "BTC/USDT"
df_list1m = {}
df_list1d = {}
df_list1h = {}
df_list5m = {}
df_list15m = {}


for pair in pair_list:
    df = get_historical_from_db(ccxt.binance(), pair, '1m', path="./database/")
    df_list1m[pair] = df.loc[:]

for pair in pair_list:
    df = get_historical_from_db(ccxt.binance(), pair, '1d', path="./database/")
    df_list1d[pair] = df.loc[:]

for pair in pair_list:
    df = get_historical_from_db(ccxt.binance(), pair, '1h', path="./database/")
    df_list1h[pair] = df.loc[:]

for pair in pair_list:
    df = get_historical_from_db(ccxt.binance(), pair, '5m', path="./database/")
    df_list5m[pair] = df.loc[:]

for pair in pair_list:
    df = get_historical_from_db(
        ccxt.binance(), pair, '15m', path="./database/")
    df_list15m[pair] = df.loc[:]
try: del(df)
except:pass
df_list = df_list1m
prerr("Data load 100% use df_list1d[\"BTC/USDT\"] for exemple to access")















'''
Binance First Candle Finders
Creslin

Get list of all IDs on binance
Returns the first candle / launch timestamp to the minute for each
'''
import urllib.request
import json
import ccxt

def all_ids():
    # load all markets from binance into a list
    id = 'binance'
    exchange_found = id in ccxt.exchanges
    if exchange_found:
        exchange = getattr(ccxt, id)({})
        markets = exchange.load_markets()
        tuples = list(ccxt.Exchange.keysort(markets).items())

        ids = []
        for (k, v) in tuples:
            ids.append(v['id'])

        return ids

def give_first_kline_open_stamp(interval, symbol, start_ts=1499990400000):
        '''
        Returns the first kline from an interval and start timestamp and symbol
        :param interval:  1w, 1d, 1m etc - the bar length to query
        :param symbol:    BTCUSDT or LTCBTC etc
        :param start_ts:  Timestamp in miliseconds to start the query from
        :return:          The first open candle timestamp
        '''

        url_stub = "http://api.binance.com/api/v1/klines?interval="

        #/api/v1/klines?interval=1m&startTime=1536349500000&symbol=ETCBNB
        addInterval   = url_stub     + str(interval) + "&"
        addStarttime  = addInterval   + "startTime="  + str(start_ts) + "&"
        addSymbol     = addStarttime + "symbol="     + str(symbol)
        url_to_get = addSymbol

        kline_data = urllib.request.urlopen(url_to_get).read().decode("utf-8")
        kline_data = json.loads(kline_data)

        return kline_data[0][0]


# Get list of all IDs on binance
def get_crypto_metadata(pair_list):
    Binance_USDT_HALAL=pair_list
    ids = []
    #ids = all_ids()
    for halalpair in Binance_USDT_HALAL:
    #    print( halalpair.replace('/',''))
        ids.append(halalpair.replace('/',''))
    #print(ids)
    MetaData=pd.DataFrame(ids)
    MetaData["Pair"]=Binance_USDT_HALAL
    counters=0
    for this_id in ids:
        '''
        Find launch Week of symbol, start at Binance launch date 2017-07-14 (1499990400000)
        Find launch Day of symbol in week
        Find launch minute of symbol in day
        '''

        symbol_launch_week_stamp   = give_first_kline_open_stamp('1w', this_id, 1499990400000 )
        symbol_launch_day_stamp    = give_first_kline_open_stamp('1d', this_id, symbol_launch_week_stamp)
        symbol_launch_minute_stamp = give_first_kline_open_stamp('1m', this_id, symbol_launch_day_stamp)
        MetaData.loc[counters,"launch_week_stamp"]=str(symbol_launch_week_stamp)
        MetaData.loc[counters,"launch_day_stamp"]=str(symbol_launch_day_stamp)
        MetaData.loc[counters,"launch_minute"]=pd.to_datetime(symbol_launch_minute_stamp, unit='ms')

        counters += 1

        #print("Week stamp", symbol_launch_week_stamp)
        #print("Day  stamp", symbol_launch_day_stamp)
        #print("Min  stamp", symbol_launch_minute_stamp)

        print(this_id, "launched", symbol_launch_minute_stamp )
    return MetaData
    #print("")
    
    
def print_pl():
    print(pair_list)
    
    
    
def check_metadata(pair_list):
    global MetaData
    MetaData = pd.read_csv("../Data/MetaData.csv",index_col=0)
    pair_list_plus=[]
    for pair in pair_list:
        if pair not in MetaData["Pair"].to_list():
            pair_list_plus.append(pair)
    if pair_list_plus:
        MetaDataPlus=get_crypto_metadata(pair_list_plus)
        MetaData.concat([MetaData,MetaDataPlus])
        MetaData.to_csv("../Data/MetaData.csv")    
    
    
    
try:
    global MetaData
    MetaData = pd.read_csv("../Data/MetaData.csv",index_col=0)
    check_metadata(Binance_USDT_HALAL)
except:
    MetaData=get_crypto_metadata(Binance_USDT_HALAL)
    MetaData.to_csv("../Data/MetaData.csv")
#allok = pd.read_csv('D:/+DATA+/allok_w15.csv')

def buy_fix(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        window=3
        #buy_pourcent=0.43
        print (f"---fixed buy--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)
        window=3
        df['ismin1'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        window=5
        df['ismin2'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        window=7
        df['ismin3'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        window=10
        df['ismin4'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        window=15
        df['ismin5'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        window=20
        df['ismin6'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df['ismin']=((df['ismin1']==1) | (df['ismin2']==1) | (df['ismin3'] == 1) |(df['ismin4']==1) |(df['ismin5'])| (df['ismin6']==1))
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---fixed buy--- no b")
    try:
        df.pop("ismin")
        df.pop("ismin1")
        df.pop("ismin2")
        df.pop("ismin3")
        df.pop("ismin4")
        df.pop("ismin5")
        df.pop("ismin6")
    except:print("---fixed buy--- no sell")

    return df

def buy_floating(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)


        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df

def buy_test(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)


        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df


def sell_test(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        print (f"---sell_test--- Sell percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo )).replace({False: 1, True: 0})'
        code=codep1+codep2
        prerr(code)
        exec(code)
    except Exception as e:
        print("Error buy only")
        print(e)
    return df






#MetaData.to_csv("D:\+DATA+\MetaData.csv")
pair_list=Binance_USDT_HALAL
window=WINDOW_SIZE
buy_weight=50
sample_size=10000
min_days=MAX_FORCAST_SIZE
buffer_size=100000
#MetaData








def expand_row(dataframe, window=2):
    df = dataframe.copy()
    for i in range(1, window+1):
        df["high"+str(i)] = df["high"][i:]
        df["low"+str(i)] = df["low"][i:]
        df["open"+str(i)] = df["open"][i:]
        df["close"+str(i)] = df["close"][i:]
        df["volume"+str(i)] = df["volume"][i:]
    return df

def justlast_remover(df):
    justlast=["BTC_open","BTC_low","BTC_close","open","low","close"]
    for key in df.keys():
        #key.find("-1") != -1 and key.find("open-1") == -1) or
        if ( key.find("close") != -1 ):
            justlast.append(key)
    df=df.drop(columns=justlast)
    return df

def expand_previous(dataframe, window=10):
    df = dataframe.copy()
    if window >= len(df):
        for i in range(1, window+1):
            df.loc[window:len(df),"high-"+str(i)]=None
            df.loc[window:len(df),"low-"+str(i)]=None
            #df.loc[window:len(df),"open-"+str(i)]=None            
            df.loc[window:len(df),"close-"+str(i)]=None            
            df.loc[window:len(df),"volume-"+str(i)]=None
        window=len(df)

    for i in range(1, window+1):
        try:
            df.loc[window:len(df),"high-"+str(i)]=None
            df["high-"+str(i)].iloc[window:len(df)]=df["high"][window-i:len(df)-i].to_list()

            df.loc[window:len(df),"low-"+str(i)]=None
            df["low-"+str(i)].iloc[window:len(df)]=df["low"][window-i:len(df)-i].to_list()

            # df.loc[window:len(df),"open-"+str(i)]=None
            # df["open-"+str(i)].iloc[window:len(df)]=df["open"][window-i:len(df)-i].to_list()           
            
            df.loc[window:len(df),"close-"+str(i)]=None
            df["close-"+str(i)].iloc[window:len(df)]=df["close"][window-i:len(df)-i].to_list()            
            
            df.loc[window:len(df),"volume-"+str(i)]=None
            df["volume-"+str(i)].iloc[window:len(df)]=df["volume"][window-i:len(df)-i].to_list()
            
            # df["high-"+str(i)][i:] = df["high"][i-1:]
            # df["low-"+str(i)][i:] = df["low"][i-1:]
            # df["open-"+str(i)][i:] = df["open"][i-1:]
            # df["close-"+str(i)][i:] = df["close"][i-1:]
            # df["volume-"+str(i)][i:] = df["volume"][i-1:]
        except:
            prerr("Error in     expand_previous: " +str(i))
    if window >= len(df): return df       
    return df.iloc[window:]

def expand_previous_org(dataframe, window=10):
    df = dataframe.copy()
    if window >= len(df):
        for i in range(1, window+1):
            df.loc[window:len(df),"high-"+str(i)]=None
            df.loc[window:len(df),"low-"+str(i)]=None
            df.loc[window:len(df),"open-"+str(i)]=None            
            df.loc[window:len(df),"close-"+str(i)]=None            
            df.loc[window:len(df),"volume-"+str(i)]=None
        window=len(df)

    for i in range(1, window+1):
        try:
            df.loc[window:len(df),"high-"+str(i)]=None
            df["high-"+str(i)].iloc[window:len(df)]=df["high"][window-i:len(df)-i].to_list()

            df.loc[window:len(df),"low-"+str(i)]=None
            df["low-"+str(i)].iloc[window:len(df)]=df["low"][window-i:len(df)-i].to_list()

            df.loc[window:len(df),"open-"+str(i)]=None
            df["open-"+str(i)].iloc[window:len(df)]=df["open"][window-i:len(df)-i].to_list()           
            
            df.loc[window:len(df),"close-"+str(i)]=None
            df["close-"+str(i)].iloc[window:len(df)]=df["close"][window-i:len(df)-i].to_list()            
            
            df.loc[window:len(df),"volume-"+str(i)]=None
            df["volume-"+str(i)].iloc[window:len(df)]=df["volume"][window-i:len(df)-i].to_list()
            
            # df["high-"+str(i)][i:] = df["high"][i-1:]
            # df["low-"+str(i)][i:] = df["low"][i-1:]
            # df["open-"+str(i)][i:] = df["open"][i-1:]
            # df["close-"+str(i)][i:] = df["close"][i-1:]
            # df["volume-"+str(i)][i:] = df["volume"][i-1:]
        except:
            prerr("Error in     expand_previous: " +str(i))
    if window >= len(df): return df       
    return df.iloc[window:]


def expand_previous_err(dataframe, window=10):
    df = dataframe.copy()
    if window >= len(df):
        for i in range(1, window+1):
            df.loc[window:len(df),"high-"+str(i)]=None
            df.loc[window:len(df),"low-"+str(i)]=None
            df.loc[window:len(df),"open-"+str(i)]=None            
            df.loc[window:len(df),"close-"+str(i)]=None            
            df.loc[window:len(df),"volume-"+str(i)]=None
        window=len(df)

    for i in range(1, window+1):
            df.loc[window:len(df),"high-"+str(i)]=None
            df["high-"+str(i)].iloc[window:len(df)]=df["high"][window-i:len(df)-i]

            df.loc[window:len(df),"low-"+str(i)]=None
            df["low-"+str(i)].iloc[window:len(df)]=df["low"][window-i:len(df)-i]

            df.loc[window:len(df),"open-"+str(i)]=None
            df["open-"+str(i)].iloc[window:len(df)]=df["open"][window-i:len(df)-i]            
            
            df.loc[window:len(df),"close-"+str(i)]=None
            df["close-"+str(i)].iloc[window:len(df)]=df["close"][window-i:len(df)-i]            
            
            df.loc[window:len(df),"volume-"+str(i)]=None
            df["volume-"+str(i)].iloc[window:len(df)]=df["volume"][window-i:len(df)-i]
            
            # df["high-"+str(i)][i:] = df["high"][i-1:]
            # df["low-"+str(i)][i:] = df["low"][i-1:]
            # df["open-"+str(i)][i:] = df["open"][i-1:]
            # df["close-"+str(i)][i:] = df["close"][i-1:]
            # df["volume-"+str(i)][i:] = df["volume"][i-1:]
    if window >= len(df):
        return df
    return df.iloc[window:]

def expand_timeframe(df_minutes,df_hours, window=2):
    dfm = df_minutes.copy()
    for j in range(1, window+1):
        for i in df_hours[dfm.iloc[0].name:].index:
        #prerr(str(i))
            try:
                dfm.loc[pd.date_range(str(i), periods=60, freq="min"),"high_1h-"+str(j)]= df_hours[str(i-pd.Timedelta(str(j)+" hour"))]['high']
                dfm.loc[pd.date_range(str(i), periods=60, freq="min"),"low_1h-"+str(j)]= df_hours[str(i-pd.Timedelta(str(j)+" hour"))]['low']
                dfm.loc[pd.date_range(str(i), periods=60, freq="min"),"open_1h-"+str(j)]= df_hours[str(i-pd.Timedelta(str(j)+" hour"))]['open']
                dfm.loc[pd.date_range(str(i), periods=60, freq="min"),"close_1h-"+str(j)]= df_hours[str(i-pd.Timedelta(str(j)+" hour"))]['close']
            except:
                prerr("Error Merging: "+str(i))
    
    return dfm


def float_or_not(var):
    try:
        x=float(var)
    except:
        x=None
    return x

def expand_to_1h(df_1m,df_1h, window=2):
    dfm = df_1m.copy()
    index_start=df_1h.index.intersection(dfm.index.round(freq='H'))
    for i in index_start:
        for j in range(1, window+1):
            # try:    
                timefragment=dfm.index.intersection(pd.date_range(str(i), periods=60, freq="min"))
                dfm.loc[timefragment,"high_1h-"+str(j)]=float_or_not(df_1h.loc[str(i-pd.Timedelta(str(j)+" hour"))]['high'])
                dfm.loc[timefragment,"low_1h-"+str(j)]=float_or_not(df_1h.loc[str(i-pd.Timedelta(str(j)+" hour"))]['low'])
                dfm.loc[timefragment,"open_1h-"+str(j)]=float_or_not(df_1h.loc[str(i-pd.Timedelta(str(j)+" hour"))]['open'])
                dfm.loc[timefragment,"close_1h-"+str(j)]=float_or_not(df_1h.loc[str(i-pd.Timedelta(str(j)+" hour"))]['close'])
            # except:
            #     prerr("error fonction "str(i))
    return dfm

def expand_to_4h(df_1m,df_4h, window=2):
    dfm = df_1m.copy()
    #index_start=df_1h[str(dfm.iloc[0].name.round(freq='H')):].index.intersection(dfm.index)
    index_start=df_4h.index.intersection(dfm.index.round(freq='4H'))
    for i in index_start:
        for j in range(1, window+1):
            # try:    
                timefragment=dfm.index.intersection(pd.date_range(str(i), periods=4*60, freq="min"))
                dfm.loc[timefragment,"high_4h-"+str(j)]=float_or_not(df_4h.loc[str(i-pd.Timedelta(str(j*4)+" hour"))]['high'])
                dfm.loc[timefragment,"low_4h-"+str(j)]= float_or_not(df_4h.loc[str(i-pd.Timedelta(str(j*4)+" hour"))]['low'])
                dfm.loc[timefragment,"open_4h-"+str(j)]= float_or_not(df_4h.loc[str(i-pd.Timedelta(str(j*4)+" hour"))]['open'])
                dfm.loc[timefragment,"close_4h-"+str(j)]= float_or_not(df_4h.loc[str(i-pd.Timedelta(str(j*4)+" hour"))]['close'])
            # except:
            #     prerr("error fonction "str(i))
    return dfm

def expand_to_1d(df_1m,df_1d, window=2,time_suffix="1d"):
    dfm = df_1m.copy()
    index_start=df_1d.index.intersection(dfm.index.round(freq='1d'))
    for i in index_start:
        for j in range(1, window+1):
            # try:    
                prerr(i)
                timefragment=dfm.index.intersection(pd.date_range(str(i), periods=24*60, freq="min"))
                dfm.loc[timefragment,"high_"+time_suffix+"-"+str(j)]=float_or_not(df_1d.loc[str(i-pd.Timedelta(str(j)+" day"))]['high'])
                dfm.loc[timefragment,"low_"+time_suffix+"-"+str(j)]= float_or_not(df_1d.loc[str(i-pd.Timedelta(str(j)+" day"))]['low'])
                dfm.loc[timefragment,"open_"+time_suffix+"-"+str(j)]= float_or_not(df_1d.loc[str(i-pd.Timedelta(str(j)+" day"))]['open'])
                dfm.loc[timefragment,"close_"+time_suffix+"-"+str(j)]= float_or_not(df_1d.loc[str(i-pd.Timedelta(str(j)+" day"))]['close'])
            # except:
            #     prerr("error fonction "str(i))
    return dfm

def expand_to_5m(df_1m,df_5m, window=2,time_suffix="5m"):
    dfm = df_1m.copy()
    index_start=df_5m.index.intersection(dfm.index.round(freq='5 min'))
    for i in index_start:
        for j in range(1, window+1):
            # try:    
                
                timefragment=dfm.index.intersection(pd.date_range(str(i), periods=5, freq="min"))
                dfm.loc[timefragment,"high_"+time_suffix+"-"+str(j)]=float_or_not(df_5m.loc[str(i-pd.Timedelta(str(j*5)+" min"))]['high'])
                dfm.loc[timefragment,"low_"+time_suffix+"-"+str(j)]= float_or_not(df_5m.loc[str(i-pd.Timedelta(str(j*5)+" min"))]['low'])
                dfm.loc[timefragment,"open_"+time_suffix+"-"+str(j)]= float_or_not(df_5m.loc[str(i-pd.Timedelta(str(j*5)+" min"))]['open'])
                dfm.loc[timefragment,"close_"+time_suffix+"-"+str(j)]= float_or_not(df_5m.loc[str(i-pd.Timedelta(str(j*5)+" min"))]['close'])
            # except:
            #     prerr("error fonction "str(i))
    return dfm


def expand_to_15m(df_1m,df_15m, window=2,time_suffix="15m"):
    dfm = df_1m.copy()
    index_start=df_15m.index.intersection(dfm.index.round(freq='5 min'))
    for i in index_start:
        for j in range(1, window+1):    
            # try:    
                timefragment=dfm.index.intersection(pd.date_range(str(i), periods=15, freq="min"))
                dfm.loc[timefragment,"high_"+time_suffix+"-"+str(j)]=float_or_not(df_15m.loc[str(i-pd.Timedelta(str(j*15)+" min"))]['high'])
                dfm.loc[timefragment,"low_"+time_suffix+"-"+str(j)]= float_or_not(df_15m.loc[str(i-pd.Timedelta(str(j*15)+" min"))]['low'])
                dfm.loc[timefragment,"open_"+time_suffix+"-"+str(j)]= float_or_not(df_15m.loc[str(i-pd.Timedelta(str(j*15)+" min"))]['open'])
                dfm.loc[timefragment,"close_"+time_suffix+"-"+str(j)]= float_or_not(df_15m.loc[str(i-pd.Timedelta(str(j*15)+" min"))]['close'])
            # except:
            #     prerr("error fonction "str(i))
    return dfm

def rapid1d_expand(df1m,df1d,window=2):
    d1min=df1m.copy()
    d1day=df1d.loc[
    d1min.index[0].round(freq='1d')-pd.Timedelta(str(window)+' day'):
    d1min.index[len(d1min)-1].round(freq='1d')+pd.Timedelta('1 day')
    ].copy()
    d1day_pre=expand_previous(d1day,window)
    d1day_pre=d1day_pre.drop(columns=['open', 'low','close','high','volume'])
    d1day_pre=d1day_pre.add_suffix("_day")
    d1min=pd.merge_asof(
        d1min, d1day_pre, on=None, left_on=None, right_on=None, left_index=True, 
        right_index=True, by=None, left_by=None, right_by=None, 
        suffixes=('', '_day'),
        tolerance=pd.Timedelta('1 day'), allow_exact_matches=True, direction='backward')
    return d1min

def rapid1h_expand(df1m,df1h,window=2):
    d1min=df1m.copy()
    d1hour=df1h.loc[
    d1min.index[0].round(freq='H')-pd.Timedelta(str(window)+' hour'):
    d1min.index[len(d1min)-1].round(freq='H')+pd.Timedelta('1 hour')
    ].copy()
    d1hour_pre=expand_previous(d1hour,window)
    d1hour_pre=d1hour_pre.drop(columns=['open', 'low','close','high','volume'])
    d1hour_pre=d1hour_pre.add_suffix("_hour")
    d1min=pd.merge_asof(
    d1min, d1hour_pre, on=None, left_on=None, right_on=None, left_index=True, 
    right_index=True, by=None, left_by=None, right_by=None, 
    suffixes=('', '_hour'),
    tolerance=pd.Timedelta('1 hour'), allow_exact_matches=True, direction='backward')
    return d1min


def rapid5m_expand(df1m,df5m,window=2):
    d1min=df1m.copy()
    d5min=df5m.loc[
    d1min.index[0].round(freq='5 min')-pd.Timedelta(str(window*5+10)+' min'):
    d1min.index[len(d1min)-1].round(freq='5 min')+pd.Timedelta('5 min')
    ].copy()
    d5min_pre=expand_previous(d5min,window)
    d5min_pre=d5min_pre.drop(columns=['open', 'low','close','high','volume'])
    d5min_pre=d5min_pre.add_suffix("_5min")
    d1min=pd.merge_asof(
    d1min, d5min_pre, on=None, left_on=None, right_on=None, left_index=True, 
    right_index=True, by=None, left_by=None, right_by=None, 
    suffixes=('', '_5min'),
    tolerance=pd.Timedelta('5 min'), allow_exact_matches=True, direction='backward')
    return d1min

def rapid15m_expand(df1m,df15m,window=2):
    d1min=df1m.copy()
    d15min=df15m.loc[
    d1min.index[0].round(freq='15 min')-pd.Timedelta(str(window*15+30)+' min'):
    d1min.index[len(d1min)-1].round(freq='15 min')+pd.Timedelta('15 min')
    ].copy()
    d15min_pre=expand_previous(d15min,window)
    d15min_pre=d15min_pre.drop(columns=['open', 'low','close','high','volume'])
    d15min_pre=d15min_pre.add_suffix("_15min")
    d1min=pd.merge_asof(
    d1min, d15min_pre, on=None, left_on=None, right_on=None, left_index=True, 
    right_index=True, by=None, left_by=None, right_by=None, 
    suffixes=('', '_15min'),
    tolerance=pd.Timedelta('15 min'), allow_exact_matches=True, direction='backward')
    return d1min


def full_expand(df1m,df5m,df15m,df1h,df1d,window=10):
    d1min=df1m.copy()
    d1min=expand_previous(d1min,window=window).drop(columns=["volume"])
    d1min=rapid1d_expand(d1min,df1d,window)
    d1min=rapid1h_expand(d1min,df1h,window)
    d1min=rapid15m_expand(d1min,df15m,window)
    d1min=rapid5m_expand(d1min,df5m,window)
    return d1min

def full_expand_org(df1m,df5m,df15m,df1h,df1d,window=10):
    d1min=df1m.copy()
    d1min=expand_previous(d1min,window=window)
    d1min=rapid1d_expand(d1min,df1d,window)
    d1min=rapid1h_expand(d1min,df1h,window)
    d1min=rapid15m_expand(d1min,df15m,window)
    d1min=rapid5m_expand(d1min,df5m,window)
    return d1min


def day_expand(data_full):
    ser = pd.to_datetime(pd.Series(data_full.index))
    data_full["day"]=ser.dt.isocalendar().day.values
    data_full["hour"]=ser.dt.hour.values
    data_full["minute"]=ser.dt.minute.values

# merging
def pair_btc(pair="LTC/USDT",window=2):
    Pair_Full=full_expand(df_list1m[pair],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[df_list1m[pair].iloc[0].name:
        df_list1m[pair].iloc[len(df_list1m[pair])-1].name],
        df_list5m["BTC/USDT"],df_list15m["BTC/USDT"],df_list1h["BTC/USDT"],df_list1d["BTC/USDT"],window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='outer',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    return Merged




def buy_results(df,min_pourcent=BUY_PERCENT):
    mino=min_pourcent*0.01
    df["buy"]=(
        ((df["high"].shift(periods=1, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| ((
          df["high"].shift(periods=2, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| ((
          df["high"].shift(periods=3, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)
    ).replace({False: 0, True: 1}) 
 
def buy_results_gen(df,min_pourcent=BUY_PERCENT,window=3):
    mino=min_pourcent*0.01
    codep1='df["buy"]=((('
    for i in range(1,window):
        codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
    codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
    code=codep1+codep2
    print(code)
    exec(code)

def buy_sell(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    mino=buy_pourcent*0.01
    maxo=-sell_pourcent*0.01
    codep1='df["buy"]=((('
    for i in range(1,window):
        codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
    codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
    code=codep1+codep2
    prerr(code)
    exec(code)
    codep1='df["sell"]=((df["buy"]==0)&(( '
    for i in range(1,window):
        codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )& (('
    codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
    code=codep1+codep2
    prerr(code)
    exec(code)
    df["bs"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

def buy_only(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    mino=buy_pourcent*0.01
    maxo=-sell_pourcent*0.01
    codep1='df["buy"]=((('
    for i in range(1,window):
        codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
    codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
    code=codep1+codep2
    prerr(code)
    exec(code)
    # codep1='df["sell"]=((df["buy"]==0)&(( '
    # for i in range(1,window):
    #     codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )& (('
    # codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
    # code=codep1+codep2
    # prerr(code)
    # exec(code)
    # df["bs"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})
    return df










def Meta_expand(data_full,metadt,pair):
    data_full["lunch_day"]=int(-(pd.to_datetime(metadt[metadt["Pair"] == pair]["launch_minute"])-pd.Timestamp('2020-01-01 00:00:00.000000')).dt.days)

def mini_expand(pair="LTC/USDT",i=0,j=10000,window=2):
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[Pair_Full.iloc[0].name-pd.Timedelta(str(window-1) +" min"):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list5m["BTC/USDT"],
        df_list15m["BTC/USDT"],
        df_list1h["BTC/USDT"],
        df_list1d["BTC/USDT"],
        window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    return Merged

def mini_expand2(pair="LTC/USDT",i=0,j=10000,window=2,metadata=MetaData):
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[Pair_Full.iloc[0].name-pd.Timedelta(str(window-1) +" min"):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list5m["BTC/USDT"],
        df_list15m["BTC/USDT"],
        df_list1h["BTC/USDT"],
        df_list1d["BTC/USDT"],
        window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    buy_sell(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=7)
    return Merged

def mini_expand3(pair="LTC/USDT",i=0,j=10000,window=2,metadata=MetaData,high_weight=3):
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" d")).round(freq='1 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list5m["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" d")).round(freq='5 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list15m["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='15 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list1h["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='1 H'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list1d["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='1 d'):Pair_Full.iloc[len(Pair_Full)-1].name],
        window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    # Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
    #         right_index=True, suffixes=('', ''))
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='inner',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    #buy_sell(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=MAX_FORCAST_SIZE)
    buy_only(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=MAX_FORCAST_SIZE)
    Merged["high"]=(Merged["open"]+high_weight*Merged["high"]+Merged["low"]+Merged["close"])/(3+high_weight)
    Merged.rename(columns={"high":"price"},inplace = True)
    Merged["BTC_high"]=(Merged["BTC_open"]+high_weight*Merged["BTC_high"]+Merged["BTC_low"]+Merged["BTC_close"])/(3+high_weight)
    Merged.rename(columns={"BTC_high":"BTC_price"},inplace = True)
    Merged=Merged.drop(columns=["BTC_open","BTC_low","BTC_close","open","low","close"])
    # Merged=justlast_remover(Merged)
    for key in Merged.keys():
        if key.find("BTC")!=-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["BTC_price"]-Merged[key])/Merged["BTC_price"]
        if key.find("BTC")==-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["price"]-Merged[key])/Merged["price"]
    return Merged

def mini_expand3old(pair="LTC/USDT",i=0,j=10000,window=2,metadata=MetaData,high_weight=3):
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" min"):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list5m["BTC/USDT"],
        df_list15m["BTC/USDT"],
        df_list1h["BTC/USDT"],
        df_list1d["BTC/USDT"],
        window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    buy_sell(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=MAX_FORCAST_SIZE)
    Merged["high"]=(Merged["open"]+high_weight*Merged["high"]+Merged["low"]+Merged["close"])/(3+high_weight)
    Merged.rename(columns={"high":"price"},inplace = True)
    Merged["BTC_high"]=(Merged["BTC_open"]+high_weight*Merged["BTC_high"]+Merged["BTC_low"]+Merged["BTC_close"])/(3+high_weight)
    Merged.rename(columns={"BTC_high":"BTC_price"},inplace = True)
    # Merged=Merged.drop(columns=["BTC_open","BTC_low","BTC_close","open","low","close"])
    Merged=justlast_remover(Merged)

    for key in Merged.keys():
        if key.find("BTC")!=-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["BTC_price"]-Merged[key])/Merged["BTC_price"]
        if key.find("BTC")==-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["price"]-Merged[key])/Merged["price"]
    return Merged



def slow_expand(pair="LTC/USDT",i=0,j=100000,window=3):
    df=mini_expand(pair=pair,i=i,j=j,window=window)
    for mx in range(1,int(len(df_list1m[pair])/j)+1) :
        df=pd.concat([df,
        mini_expand(pair=pair,
        i=(mx*j)-window,
        j=(mx+1)*j,
        window=window)],axis=0)
    return df

def pair_data_gen(pair="LTC/USDT",i=0,j=100000,window=3,metadata=MetaData):
    df=mini_expand3(pair=pair,i=i,j=j,window=window,metadata=metadata)
    for mx in range(1,int(len(df_list1m[pair])/j)+1) :
        df=pd.concat([df,
        mini_expand3(pair=pair,
        i=(mx*j)-window,
        j=(mx+1)*j,
        window=window,metadata=metadata)],axis=0)
        # Meta_expand(df,metadata,pair)
        # buy_sell(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=7)
        #print("loop "+str(mx)+"--> size of df: "+str(len(df)))
    return df

###

def data_is_enough(df,days=10,window=10):
    if days <= window:
        return df[~df.isnull().any(axis=1) |(df["open-"+str(days)+"_day"].isnull() & ~df["open-"+str(window-1)+"_hour"].isnull()  & ~df["open-"+str(window-1)+"_5min"].isnull() & ~df["open-"+str(days-1)+"_day"].isnull())]
    else:
        prerr("number of days must be equal or lower than window")
        return df

def data_cleanup(df):
    return df.dropna()
    #return df[~df.isnull().any(axis=1)]
    
def data_shufler2(df):
    return df.sample(frac=1).reset_index()
    
def data_shufler(df):
    x = len(df)
    df["num_index"] = range(0, x, 1)
    df.set_index(df['num_index'], inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    try:df= df.drop("num_index",axis=1)
    except:pass
    #df = df.reindex(np.random.permutation(df.index))
    return df
    
def data_np_shufler(df):
    df = shuffle(df)
    #df = df.reindex(np.random.permutation(df.index))
        



def data_chooser(df,weight=50,row_numbers=100000):
    df=data_shufler(df)
    if row_numbers>=len(df):
        row_numbers=len(df)
    df=pd.concat([df[df["buy"]==1].iloc[:int(row_numbers*weight*0.01)],
                 df[df["buy"]==0].iloc[:int(row_numbers*(100-weight)*0.01)]])
    df=data_shufler(df)
    return df
    
    
#from imblearn.over_sampling import RandomOverSampler
def data_chooser50(df,row_numbers=100000):
    
    df=data_shufler(df)
    if row_numbers>=len(df):
        row_numbers=len(df)
    halfrows=int(row_numbers/2)
    dfbuy=df[df["buy"]==1].iloc[:halfrows]
    while(dfbuy.shape[0]<halfrows):
        dfbuy=pd.concat([dfbuy,dfbuy]).iloc[:halfrows]
    dfnob=df[df["buy"]==0].iloc[:halfrows]
    df=pd.concat([dfbuy,
                 dfnob])
    df=data_shufler(df)
    return df

def data_looper(pair_list=Binance_USDT_HALAL,window=15,buy_weight=50,sample_size=100000,min_days=10,buffer_size=100000):
    xdf=pd.DataFrame()
    for pair in pair_list:
        if pair != "BTC/USDT":
            print("working on: "+pair)
            df=pair_data_gen(pair=pair,i=0,j=buffer_size,window=window)
            gc.collect()

            df=data_is_enough(df,days=min_days,window=window)
            gc.collect()

            df=data_chooser(df,weight=buy_weight,row_numbers=sample_size)
            gc.collect()

            print(pair+" is processed")
            xdf=pd.concat([xdf,df],axis=0)
            del(df)
            gc.collect()
        else:
            print("ignore BTC")
    return xdf

def data_looper_fast(pair_list=Binance_USDT_HALAL,window=15,buy_weight=50,sample_size=100000,min_days=10,buffer_size=100000):
    xdf=pd.DataFrame()
    count=0
    for pair in pair_list:
        if pair != "BTC/USDT":
            print("working on: "+pair ,end=" -->")
            try:
                df=pair_data_gen(pair=pair,i=0,j=buffer_size,window=window)
                gc.collect()
                count+=1
                # df=data_is_enough(df,days=min_days,window=window)
                # gc.collect()
                df.reindex(np.random.permutation(df.index))
                df=data_chooser(df,weight=buy_weight,row_numbers=sample_size)
                gc.collect()

                df=data_cleanup(df)
                print(pair+" is processed")
            except:
                print(f"error while processing {pair} {count}/{len(pair_list)}")
            xdf=pd.concat([xdf,df],axis=0)
            del(df)
            gc.collect()
        else:
            print("ignore BTC")
            
    return xdf

def volume_cleaner(df):
    VolRemover=["volume","volume-1","BTC_volume","BTC_volume-1"]
    for key in df.keys():
        if key.find("volume-1_") != -1 :
            VolRemover.append(key)
    df=df.drop(columns=VolRemover)
    return df
    
Normalization=None
def normalize(dataset,file=Normalization_File):
    global Normalization
    try:
        N=Normalization
    except:
        Normalization=None
    if(Normalization==None):
        #print('Loading normalization from file')
        with open(file) as json_file:
            Normalization = json.load(json_file)
    else:
        #print('normalization is loaded')
        pass

    mean=np.array(Normalization["mean"])
    std=np.array(Normalization["std"])
    dataset -= mean 
    dataset /= std
    return(dataset)















# Costumaize buy condition here
def buy_only(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        ## test param
        #buy_pourcent=1
        window=15
        print (f"---buy_only--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        # codep1='df["b"]=((('
        # for i in range(1,window):
        #     codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        # codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        # code=codep1+codep2
        # prerr(code)
        # exec(code)


##################### debug ###############33
        strcomment='''
        codep1='df["b5"]=((('
        for i in range(1,5):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)



        codep1='df["b10"]=((('
        for i in range(1,10):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        codep1='df["b15"]=((('
        for i in range(1,15):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)


        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["b30"]=((('
        for i in range(1,30):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)


        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["b60"]=((('
        for i in range(1,60):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["b180"]=((('
        for i in range(1,180):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)
    
        '''#####

        # codep1='df["b5p"]=((('
        # for i in range(1,window+5):
        #     codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.005 )| (('
        # codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.005)).replace({False: 0, True: 1})'
        # code=codep1+codep2
        # prerr(code)
        # exec(code)

        # codep1='df["b9p"]=((('
        # for i in range(1,window+9):
        #     codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.009 )| (('
        # codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.009)).replace({False: 0, True: 1})'
        # code=codep1+codep2
        # prerr(code)
        # exec(code)

        # codep1='df["b15p"]=((('
        # for i in range(1,window+15):
        #     codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.015 )| (('
        # codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=0.015)).replace({False: 0, True: 1})'
        # code=codep1+codep2
        # prerr(code)
        # exec(code)






        # codep1='df["sell"]=((( '
        # for i in range(1,window):
        #     codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )& (('
        # codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        # code=codep1+codep2
        # prerr(code)
        # exec(code)

        # df["buy"]=((df['b']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})
        
        




        #df.pop("b")
        #df.pop("sell")
        
        
        # df["f1"]=(df["high"].shift(periods=-1, freq=None, axis=0, fill_value=None)-df["high"])/df["high"]
        # df["f2"]=(df["high"].shift(periods=-2, freq=None, axis=0, fill_value=None)-df["high"])/df["high"]
        # df["f15_p"]=df["high-1_15min"].shift(periods=-2, freq=None, axis=0, fill_value=None)
        # df["fh_p"]=df["high-1_hour"].shift(periods=-2, freq=None, axis=0, fill_value=None)

        # df['signal'] = np.where(
        # df['high'] >= df.shift(1).rolling(window)['high'].min(), 1,
        # np.where(
        #     df['low'] >= df.shift(1).rolling(window)['low'].min(), -1,
        #     0
        # )
        # )




    #     df['ismin'] = np.where(
    #     df['high'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
    #    0
    #     )
    #     df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})
    #     df.pop("b")
    #     df.pop("ismin")

        

    #     df['ismax'] = np.where(
    #     df['low'] >= df.shift(1).rolling(window)['low'].min(), 1,
    #    0
    #     )
        
        
        codep1='df["buy"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )& (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        #df["buy"]=((df['b']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})
        
        

    except Exception as e:
        print("Error buy only")
        print(e)
    # p=df.pop("b")
    # p=df.pop("sell")

    return df

# Costumaize buy condition here
def buy_up_only(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        ## test param
        #buy_pourcent=1
        #window=15
        print (f"---buy_only--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["b"]=((('
        #codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['b']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})
        
        

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("sell")
    except:print("---buy_only--- no sell")

    return df


# Costumaize buy condition here
def buy_up_only(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        ## test param
        #buy_pourcent=1
        #window=15
        print (f"---buy_up_only--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["b"]=((('
        #codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['b']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})
        
        

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("sell")
    except:print("---buy_only--- no sell")

    return df


# Costumaize buy condition here
def buy_up(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("sell")
    except:print("---buy_only--- no sell")

    return df

def buy_after_depth_close(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["close"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["close"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['close'].shift(1) <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})
        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df


def buy_after_depth_closeV1(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["close"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["close"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['close'].shift(1) <= df.shift(-window-1).rolling(2*window)['close'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})
        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df

def buy_up_close(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["close"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["close"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("sell")
    except:print("---buy_only--- no sell")

    return df

def buy_up_close2(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["close"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["close"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("sell")
    except:print("---buy_only--- no sell")

    return df



# Costumaize buy condition here
def buy_min_up(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['high'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("ismin")
    except:print("---buy_only--- no sell")

    return df

def buy_min_close(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["close"])/df["close"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['close'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("ismin")
    except:print("---buy_only--- no sell")

    return df


def buy_after_depth(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['high'].shift(1) <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})
        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df

def buy_after_depth2(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_after_depth--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )& (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['high'].shift(1) <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})
        
        codep1='df["sell"]=((( '
        for i in range(1,window):
            codep1=codep1+'df["low"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo ) | (('
        codep2='df["low"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] <=maxo )).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df["buy"]=((df['buy']==1 ) & (df['sell']==0)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_after_depth--- no b")
    try:df.pop("ismin")
    except:print("---buy_after_depth--- no sell")
    try:df.pop("sell")
    except:print("---buy_after_depth--- no sell")
    return df

def buy_the_dip(df,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy percent: {buy_pourcent}% MaxForcastSize: {window}")
        mino=buy_pourcent*0.01
        maxo=-sell_pourcent*0.01
        codep1='df["buy"]=((('
        for i in range(1,window):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        df['ismin'] = np.where(
        df['high'] <= df.shift(-window-1).rolling(2*window)['high'].min(), 1,
       0
        )
        df["buy"]=((df['buy']==1 ) & (df['ismin']==1)).replace({False: 0, True: 1})

    except Exception as e:
        print("Error buy only")
        print(e)
    try:df.pop("b")
    except:print("---buy_only--- no b")
    try:df.pop("ismin")
    except:print("---buy_only--- no sell")

    return df

def mini_expand4(pair="GMT/USDT",i=0,j=10000,window=2,metadata=MetaData,high_weight=1,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,buy_function=buy_min_up):
    print(f"mini_expand : {pair}")
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    BTC_Full=full_expand(
        df_list1m["BTC/USDT"].loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" d")).round(freq='1 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list5m["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" d")).round(freq='5 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list15m["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='15 min'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list1h["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='1 H'):Pair_Full.iloc[len(Pair_Full)-1].name],
        df_list1d["BTC/USDT"],#.loc[(Pair_Full.iloc[0].name-pd.Timedelta(str(window) +" day")).round(freq='1 d'):Pair_Full.iloc[len(Pair_Full)-1].name],
        window)   
    BTC_Full=BTC_Full.add_prefix("BTC_")
    # Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
    #         right_index=True, suffixes=('', ''))
    Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='inner',
            right_index=True, suffixes=('', ''))
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    #buy_sell(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=MAX_FORCAST_SIZE)
    buy_function(Merged,buy_pourcent=buy_pourcent,sell_pourcent=sell_pourcent,window=MAX_FORCAST_SIZE)
    Merged["high"]=(Merged["open"]+high_weight*Merged["high"]+Merged["low"]+Merged["close"])/(3+high_weight)
    Merged.rename(columns={"high":"price"},inplace = True)
    Merged["BTC_high"]=(Merged["BTC_open"]+high_weight*Merged["BTC_high"]+Merged["BTC_low"]+Merged["BTC_close"])/(3+high_weight)
    Merged.rename(columns={"BTC_high":"BTC_price"},inplace = True)
    Merged=Merged.drop(columns=["BTC_open","BTC_low","BTC_close","open","low","close"])
    # Merged=justlast_remover(Merged)
    for key in Merged.keys():
        if key.find("BTC")!=-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["BTC_price"]-Merged[key])/Merged["BTC_price"]
        if key.find("BTC")==-1 and (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["price"]-Merged[key])/Merged["price"]
    Merged=Merged.dropna()
    return Merged

def mini_expand4_btc(i=0,j=10000,window=2,metadata=MetaData,high_weight=1,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT):
    pair="BTC/USDT"
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    
    # Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
    #         right_index=True, suffixes=('', ''))
    Merged=Pair_Full
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    #buy_sell(Merged,buy_pourcent=BUY_PERCENT,sell_pourcent=SELL_PERCENT,window=MAX_FORCAST_SIZE)
    buy_function(Merged,buy_pourcent=buy_pourcent,sell_pourcent=sell_pourcent,window=MAX_FORCAST_SIZE)
    Merged["high"]=(Merged["open"]+high_weight*Merged["high"]+Merged["low"]+Merged["close"])/(3+high_weight)
    Merged.rename(columns={"high":"price"},inplace = True)

    Merged=Merged.drop(columns=["open","low","close"])
    # Merged=justlast_remover(Merged)
    for key in Merged.keys():
        if  (key.find("open")!=-1 or
        key.find("high")!=-1 or key.find("low")!=-1 or key.find("close")!=-1):
            Merged[key]=(Merged["price"]-Merged[key])/Merged["price"]
    Merged=Merged.dropna()
    return Merged

def data_chooser50(df,row_numbers=100000):
    
    df=data_shufler(df)
    if row_numbers>=len(df):
        row_numbers=len(df)
    halfrows=int(row_numbers/2)
    dfbuy=df[df["buy"]==1].iloc[:halfrows]
    while(dfbuy.shape[0]<halfrows):
        dfbuy=pd.concat([dfbuy,dfbuy]).iloc[:halfrows]
    dfnob=df[df["buy"]==0].iloc[:halfrows]
    df=pd.concat([dfbuy,
                 dfnob])
    df=data_shufler(df)
    return df


def human_percent(float_percent,type_string="Precent Mean",ShowMessage=True):
    nb=round(float_percent*100,3)
    if ShowMessage: print(type_string+": "+"{:.3f}".format(nb)+"%")
    return nb
hp=human_percent


def fixdt(dt):
    dt=np.nan_to_num(dt,nan=0)
    dt=np.nan_to_num(dt, neginf=0) 
    dt=np.nan_to_num(dt, posinf=0) 
    dt=dt.astype(np.float32)
    return dt

def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return list(intersection)