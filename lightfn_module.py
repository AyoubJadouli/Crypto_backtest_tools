
import gc
from collections import defaultdict
from functools import partial
import ccxt
import json
import os
#################### Files #################

DATABASE_DIR="./database"
INFO_DIR=f"{DATABASE_DIR}/Info"
ALL_BINANCE_TICKERS_LISTFILE=f"{INFO_DIR}/ALL_BINANCE_TICKERS_LISTFILE.json"
ONLY_MY_HALAL_LIST_FILE=f"{INFO_DIR}/ONLY_MY_HALAL_LIST.json"
halal_list_file_path=f"{INFO_DIR}/halal-crypto-list.txt"
BINANCE_KLINES_DATA_DIR=f"{DATABASE_DIR}/OpenBinance/KLINES"
BINANCE_TRADES_DATA_DIR=f"{DATABASE_DIR}/OpenBinance/Trades"
METADATA_FILE=f"{INFO_DIR}/METADATA.csv"
COINGEKO_INFO_FILE=f"{INFO_DIR}/COINGEKO_INFO_FILE.csv"
########### data info ################
USE_TRAILING_STOP_LOSS= False
### Data functions

def get_all_binance_tickers():
    """Return a list of all binance tickers from a local file or an API call."""
    try:
        with open(ALL_BINANCE_TICKERS_LISTFILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        url = 'https://api.binance.com/api/v3/ticker/price'
        response = requests.get(url)
        tickers = response.json()
        ticker_list = [ticker['symbol'] for ticker in tickers]
        with open(ALL_BINANCE_TICKERS_LISTFILE, "w") as f:
            json.dump(ticker_list, f)
        return ticker_list

    
def get_my_halal_list(halal_file=halal_list_file_path,ticker_list = []):
    
    try:
        with open(ONLY_MY_HALAL_LIST_FILE, "r") as f:
            json.dump(MY_HALA_DIC, f)
        
        VOLATILE_USDT_PAIRS=MY_HALA_DIC["USDT_PAIRS"]
        VOLATILE_BUSD_PAIRS=MY_HALA_DIC["BUSD_PAIRS"]
    except:
    
        ticker_list = ALL_BINANCE_TICKERS
        
        with open(halal_file, "r") as f:
            VOLATILE_COINS = [line.strip() for line in f.readlines()]
        
        VOLATILE_USDT_PAIRS = [f"{coin}/USDT" for coin in VOLATILE_COINS]
        VOLATILE_BUSD_PAIRS = [f"{coin}/BUSD" for coin in VOLATILE_COINS]
        
        # Remove BUSD pairs not listed in Binance
        VOLATILE_BUSD_PAIRS = [pair for pair in VOLATILE_BUSD_PAIRS if pair.replace('/', '') in ticker_list]
        
        # Remove USDT pairs not listed in Binance
        VOLATILE_USDT_PAIRS = [pair for pair in VOLATILE_USDT_PAIRS if pair.replace('/', '') in ticker_list]
        
        # # Remove USDT pairs that don't have 1m data
        # content = os.listdir('database/DataBackTest/1m')
        # VOLATILE_USDT_PAIRS = [pair for pair in VOLATILE_USDT_PAIRS if f"{pair.replace('/', '-')}.csv" in content]
        
        # # Remove BUSD pairs that don't have 1m data
        # VOLATILE_BUSD_PAIRS = [pair for pair in VOLATILE_BUSD_PAIRS if f"{pair.replace('/', '-')}.csv" in content]
    MY_HALA_DIC={"WORKING_PAIRS":VOLATILE_USDT_PAIRS+VOLATILE_BUSD_PAIRS,"VOLATILE_USDT_PAIRS":VOLATILE_USDT_PAIRS,"VOLATILE_BUSD_PAIRS":VOLATILE_BUSD_PAIRS}
    with open(ONLY_MY_HALAL_LIST_FILE, "w") as f:
        json.dump(MY_HALA_DIC, f)
    return VOLATILE_USDT_PAIRS + VOLATILE_BUSD_PAIRS , VOLATILE_USDT_PAIRS ,VOLATILE_BUSD_PAIRS


def extract_pairs(list):
    result = []
    for element in list:
        if element.endswith('USDT') or element.endswith('BUSD'):
            pair = element[:-4] + '/' + element[-4:]
            result.append(pair)
    return result



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
    MetaData = pd.read_csv(METADATA_FILE,index_col=0)
    pair_list_plus=[]
    for pair in pair_list:
        if pair not in MetaData["Pair"].to_list():
            pair_list_plus.append(pair)
    if pair_list_plus:
        MetaDataPlus=get_crypto_metadata(pair_list_plus)
        MetaData.concat([MetaData,MetaDataPlus])
        MetaData.to_csv(METADATA_FILE)    
    

 
def check_metadata(pair_list):
    global MetaData
    MetaData = pd.read_csv(METADATA_FILE,index_col=0)
    pair_list_plus=[]
    for pair in pair_list:
        if pair not in MetaData["Pair"].to_list():
            pair_list_plus.append(pair)
    if pair_list_plus:
        MetaDataPlus=get_crypto_metadata(pair_list_plus)
        MetaData.concat([MetaData,MetaDataPlus])
        MetaData.to_csv(METADATA_FILE)    
    

def read_metadata(pairs):
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
    try:
        MetaData = pd.read_csv(METADATA_FILE,index_col=0)
        pairs_in_csv = set(MetaData['Pair'].tolist())
        pairs_to_get_metadata = list(set(pairs) - pairs_in_csv)
        if pairs_to_get_metadata:
            new_metadata = get_crypto_metadata(pairs_to_get_metadata)
            MetaData = pd.concat([MetaData, new_metadata], ignore_index=True)
            MetaData.to_csv(METADATA_FILE, index=False)
        return MetaData
    except:
        MetaData = get_crypto_metadata(pairs)
        MetaData.to_csv(METADATA_FILE, index=False)
        return MetaData
    
    
    
### Data Work






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



def get_historical_dataframes(working_pairs, oldest_pair="BTC/USDT", timeframes=("1m", "1d", "1h", "5m", "15m"), path="./database/"):
    if oldest_pair not in working_pairs:
        working_pairs.append(oldest_pair)
    
    error_list = []
    dataframes = defaultdict(dict)

    binance = ccxt.binance()
    get_backtest_historical_partial = partial(get_backtest_historical, binance, path=path)

    for pair in working_pairs:
        for tf in timeframes:
            try:
                df = get_backtest_historical_partial(pair, tf)
                dataframes[tf][pair] = df.loc[:]
                del(df)
            except Exception as e:
                print(f" error on {pair} : {e} ")
                error_list.append(pair)
                break

    gc.collect()
    print("Data load 100% use dataframes['1d']['BTC/USDT'] for example to access")
    return dataframes, error_list

