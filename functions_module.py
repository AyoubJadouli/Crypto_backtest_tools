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

#from utilities.backtesting import plot_wallet_vs_asset, get_metrics, get_n_columns, basic_multi_asset_backtest, plot_sharpe_evolution, plot_bar_by_month
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
    

def read_metadata(pairs):
    if not os.path.exists("../Data"):
        os.makedirs("../Data")
    try:
        MetaData = pd.read_csv("../Data/MetaData.csv",index_col=0)
        pairs_in_csv = set(MetaData['Pair'].tolist())
        pairs_to_get_metadata = list(set(pairs) - pairs_in_csv)
        if pairs_to_get_metadata:
            new_metadata = get_crypto_metadata(pairs_to_get_metadata)
            MetaData = pd.concat([MetaData, new_metadata], ignore_index=True)
            MetaData.to_csv("../Data/MetaData.csv", index=False)
        return MetaData
    except:
        MetaData = get_crypto_metadata(pairs)
        MetaData.to_csv("../Data/MetaData.csv", index=False)
        return MetaData
    
def extract_pairs(list):
    result = []
    for element in list:
        if element.endswith('USDT') or element.endswith('BUSD'):
            pair = element[:-4] + '/' + element[-4:]
            result.append(pair)
    return result
    
try:
    global MetaData
    MetaData = pd.read_csv("../Data/MetaData.csv",index_col=0)
    check_metadata(Binance_USDT_HALAL)
except:
    MetaData=get_crypto_metadata(Binance_USDT_HALAL)
    MetaData.to_csv("../Data/MetaData.csv")
#allok = pd.read_csv('D:/+DATA+/allok_w15.csv')

def buy_fix(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        try:
            window=MAX_FORCAST_SIZE
            print(f"fied buy window={MAX_FORCAST_SIZE}")
        except:
            window=3
            print("fied buy window=3")
        #BUY_PCT=0.43
        print (f"---fixed buy--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_floating(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_test(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_test2(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    max_forecast_size=window#MAX_FORCAST_SIZE
    after_dip_val=1
    print(f"ganerating test point with forcast size={max_forecast_size} at {BUY_PCT}% of the current price  ...")
    mino = BUY_PCT / 100.0    
    rolling_max_close_diff = ((df['close'].rolling(window=window).max().shift(-window+1) / df['close']) - 1).fillna(0)
    df['buy']=(rolling_max_close_diff >= mino).astype(int)
    return df

def buy_alwase(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    df['buy']=1
    return df


def sell_test(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        print (f"---sell_test--- Sell pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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




def buy_results(df,min_pct=BUY_PCT):
    mino=min_pct*0.01
    df["buy"]=(
        ((df["high"].shift(periods=1, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| ((
          df["high"].shift(periods=2, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| ((
          df["high"].shift(periods=3, freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)
    ).replace({False: 0, True: 1}) 
 
def buy_results_gen(df,min_pct=BUY_PCT,window=3):
    mino=min_pct*0.01
    codep1='df["buy"]=((('
    for i in range(1,window):
        codep1=codep1+'df["high"].shift(periods='+str(i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
    codep2='df["high"].shift(periods='+str(window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
    code=codep1+codep2
    print(code)
    exec(code)

def buy_sell(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    mino=BUY_PCT*0.01
    maxo=-SELL_PCT*0.01
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

def buy_only(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    mino=BUY_PCT*0.01
    maxo=-SELL_PCT*0.01
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
    buy_sell(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=7)
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
    #buy_sell(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
    buy_only(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
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
    buy_sell(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
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
        # buy_sell(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=7)
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
def buy_only(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        ## test param
        #BUY_PCT=1
        window=15
        print (f"---buy_only--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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


        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
        codep1='df["b30"]=((('
        for i in range(1,30):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)


        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
        codep1='df["b60"]=((('
        for i in range(1,60):
            codep1=codep1+'df["high"].shift(periods='+str(-i)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino )| (('
        codep2='df["high"].shift(periods='+str(-window)+', freq=None, axis=0, fill_value=None)-df["high"])/df["high"] >=mino)).replace({False: 0, True: 1})'
        code=codep1+codep2
        prerr(code)
        exec(code)

        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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
def buy_up_only(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        ## test param
        #BUY_PCT=1
        #window=15
        print (f"---buy_only--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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
def buy_up_only(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        ## test param
        #BUY_PCT=1
        #window=15
        print (f"---buy_up_only--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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
def buy_up(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_after_depth_close(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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


def buy_after_depth_closeV1(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_up_close(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_up_close2(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_simple_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        print (f"---buy_only--- Max time window: {window}%")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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
def buy_min_up(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_min_close(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

# def buy_test2(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
#     max_forecast_size=window#MAX_FORCAST_SIZE
#     after_dip_val=1
#     print(f"optimalbuy buy maximum forcast size={max_forecast_size} at {BUY_PCT}% of the current price ")
#     mino = BUY_PCT / 100.0    
#     rolling_max_close_diff = ((df['close'].rolling(window=window).max().shift(-window+1) / df['close']) - 1).fillna(0)
#     df['buy']=(rolling_max_close_diff >= mino).astype(int)
    return df

def buy_optimal(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE):
    #df = df.fillna(0)
    mino = BUY_PCT / 100.0
    maxo = SELL_PCT / 100.0
    
    max_forecast_size=window#MAX_FORCAST_SIZE
    try:
        after_dip_val=AFTER_MARK
    except:
        after_dip_val=1
    print(f"after mark = : {after_dip_val}")
    try:
        print(f"optimalbuy buy maximum forcast size={max_forecast_size} at {BUY_PCT}% of the current price ")
    except:
        max_forecast_size = 3
        print("optimalbuy buy default window=3")
        
    rolling_max_close_diff = ((df['close'].rolling(window=window).max().shift(-window+1) / df['close']) - 1).fillna(0)
    df['buy']=(rolling_max_close_diff >= mino).astype(int)
    
    # Compute rolling minimum values
    
    window_list=[7,window]#[3, 5, 7, 10, 15, 20]
    
    for window_size in window_list:
        col_name = f'ismin{window_size}'
        rolling_min = (df['close'].shift(after_dip_val) <= df.shift(-window_size-1)['close'].rolling(2*window_size).min())
        df = df.assign(**{col_name: rolling_min.astype(int)})

    df['ismin'] = df[[f'ismin{window_size}' for window_size in window_list ]].any(axis=1).astype(int)        

    # # Compute buy and sell signals
    rolling_low_close_diff =  ((df['low'].rolling(window=int(window/2)).min().shift(-int(window/2)+1)/ df['close'] ) -1).fillna(0)
    df['sell'] = (rolling_low_close_diff <= -maxo).astype(int)

    
    # Compute final buy signal
    df['buy'] = ((df['buy'] == 1) & (df['sell'] == 0) & (df['ismin'] == 1)).astype(int)
    # Remove unnecessary columns
    df = df.drop(columns=['sell', 'ismin'] + [f'ismin{window_size}' for window_size in window_list], errors='ignore')
    return df

def buy_after_depth(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_after_depth2(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_after_depth--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def buy_the_dip(df,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=3):
    try:
        
        print (f"---buy_min_up--- Buy pct: {BUY_PCT}% MaxForcastSize: {window}")
        mino=BUY_PCT*0.01
        maxo=-SELL_PCT*0.01
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

def mini_expand4(pair="GMT/USDT",i=0,j=10000,window=2,metadata=MetaData,high_weight=1,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,buy_function=buy_min_up):
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
    #buy_sell(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
    buy_function(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
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

    meta_expend=Meta_expand
def mini_expand4_btc(i=0,j=10000,window=2,metadata=MetaData,high_weight=1,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT):
    pair="BTC/USDT"
    Pair_Full=full_expand(df_list1m[pair].iloc[i:j],df_list5m[pair],df_list15m[pair],df_list1h[pair],df_list1d[pair],window)
    
    # Merged=pd.merge(Pair_Full, BTC_Full, left_index=True, how='left',
    #         right_index=True, suffixes=('', ''))
    Merged=Pair_Full
    day_expand(Merged)
    Meta_expand(Merged,metadata,pair)
    #buy_sell(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
    buy_function(Merged,BUY_PCT=BUY_PCT,SELL_PCT=SELL_PCT,window=MAX_FORCAST_SIZE)
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

def mini_expand5(pair="GMT/USDT", i=0, j=10000, window=2, metadata=MetaData,
                 high_weight=1, BUY_PCT=BUY_PCT, SELL_PCT=SELL_PCT,
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
    print(f"mini_expand : {pair}")
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
    print(f'######################  mini_expand5 {pair} - shape {merged.shape}  buy mean : {hp(merged.buy.mean())} ############################')
    return merged

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


def human_pct(float_pct,type_string="Precent Mean",ShowMessage=True):
    nb=round(float_pct*100,3)
    if ShowMessage: print(type_string+": "+"{:.3f}".format(nb)+"%")
    return nb
hp=human_pct


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

def find_outersection(list1, list2):
    """
    Finds the outersection of two lists, i.e. the items that appear in only one of the lists.
    
    Args:
        list1 (list): The first list.
        list2 (list): The second list.
    
    Returns:
        list: The outersection of the two lists.
    """
    intersection = set(list1).intersection(set(list2))
    return list(filter(lambda x: x not in intersection, list1 + list2))


import requests

url = 'https://api.binance.com/api/v3/ticker/price'

response = requests.get(url)
tickers = response.json()
ticker_list=[]
for ticker in tickers:
    ticker_list.append((ticker['symbol']))

ALL_BINANCE_TICKERS=ticker_list



def add_technical_indicators(df):
    # Calculate indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA100'] = df['Close'].rolling(100).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD Hist'] = df['MACD'] - df['Signal Line']
    
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    df['Upper Band'], df['Middle Band'], df['Lower Band'] = ta.volatility.bollinger_bands(
        df['Close'], window=20, window_dev=2)
    
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Normalize all values between -1 and 1
    df = (df - df.mean()) / df.std()
    
    return df

## Ploter
import matplotlib.pyplot as plt

def plot_data(Model_FileName, pair_to_test, winratio, OnePair_DF, i_start, window_size, PREDICTION_TO_TEST,dot_color="g"):
    i_end=i_start+window_size
    mname = Model_FileName.replace("/UltimeTradingBot/Data","")
    coin = pair_to_test.replace('/', '-')
    mtitle = f"{coin} WinRatio:{hp(winratio)}% - {mname}".replace("/", "-")
    
    x = np.linspace(0, 10, 500)
    dashes = []  # 10 points on, 5 off, 100 on, 5 off
    fig, ax = plt.subplots()
    line1, = ax.plot(OnePair_DF.index[i_start:i_end], OnePair_DF.price[i_start:i_end], '-', linewidth=1,
                 label='price',c="w")
    line1.set_dashes(dashes)
    plt.plot(OnePair_DF[i_start:i_end][PREDICTION_TO_TEST[i_start:i_end]==1].index, OnePair_DF[i_start:i_end][PREDICTION_TO_TEST[i_start:i_end]==1].price, 'ro',c=dot_color,markersize=5)
    plt.title(mtitle)
    
    plt.show()



# Import necessary libraries
import pandas as pd
import mplfinance as mpf

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

def get_closest_index(df, date):
    idx = df.index.searchsorted(date)
    if idx >= len(df):
        return len(df) - 1
    elif idx == 0:
        return 0
    else:
        before = df.index[idx - 1]
        after = df.index[idx]
        if after - date < date - before:
            return idx
        else:
            return idx - 1


def mini_expand6(pair="GMT/USDT", i=0, j=10000, window=2, metadata=MetaData,
                 high_weight=1, BUY_PCT=BUY_PCT, SELL_PCT=SELL_PCT,
                 buy_function=buy_alwase):
    start_index=i
    end_index=j
    window_size=window
    buy_fn=buy_function    
    print(f"mini_expand : {pair}")
    pair_df = df_list1m[pair][start_index:end_index]
    btc_df = df_list1m["BTC/USDT"].loc[(pair_df.index[0] - pd.DateOffset(days=window_size+1)).round(freq='1 min'):pair_df.index[-1]+pd.Timedelta(f"{window_size} day")]
    pair_full = full_expand(pair_df, df_list5m[pair], df_list15m[pair], df_list1h[pair], df_list1d[pair], window_size)
    btc_full = full_expand(btc_df, df_list5m["BTC/USDT"], df_list15m["BTC/USDT"], df_list1h["BTC/USDT"], df_list1d["BTC/USDT"], window_size)   
    btc_full = btc_full.add_prefix("BTC_")
    merged = pd.concat([pair_full, btc_full], axis=1)
    day_expand(merged)
    Meta_expand(merged, metadata, pair)
    merged = buy_fn(merged, BUY_PCT=BUY_PCT, SELL_PCT=SELL_PCT, window=MAX_FORCAST_SIZE)
    merged["high"] = (merged["open"] + high_weight * merged["high"] + merged["low"] + merged["close"]) / (3 + high_weight)
    merged["BTC_high"] = (merged["BTC_open"] + high_weight * merged["BTC_high"] + merged["BTC_low"] + merged["BTC_close"]) / (3 + high_weight)
    merged.rename(columns={"high":"price", "BTC_high":"BTC_price"}, inplace=True)
    merged = merged.drop(columns=["BTC_open","BTC_low","BTC_close","open","low","close"])
    btc_price = merged["BTC_price"]
    merged = merged.filter(regex="(BTC|open|high|low|close)")
    merged = (btc_price - merged) / btc_price
    merged = merged.dropna()
    print(f'######################  mini_expand5 {pair} - shape {merged.shape}  buy mean : {hp(merged.buy.mean())} ############################')
    return merged




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
        
        
#!pip install psutil  # install psutil package if not already installed

import psutil

# Get the memory usage of the Jupyter Notebook process
process = psutil.Process()
memory_info = process.memory_info()

# Print the memory usage in MB
print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
