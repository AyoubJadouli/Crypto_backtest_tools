#sys.path.append('/UltimeTradingBot/Crypto_backtest_tools')
WINDOW_SIZE=10
MAX_FORCAST_SIZE=5
BUY_PCT=0.5
BUY_MODE="BUY_OPTIMAL" #"BUY_FIX" #"BUY_TEST" #"BUY_MIN_CLOSE" # "BUY_TEST" # "BUY_UP_CLOSE" #"BUY_UP" #"BUY_ONLY" #"BUY_DIP" #AFTER_DEPTH #"BUY_UP_CLOSE" #"AFTER_DEPTH_CLOSE"
SELL_PCT=0.2
AFTER_MARK=1
VERSION=5
TESTING_MOD=False
UPGRAD_MOD=False
JUST_IMPORT_DATA=False
#Normalization_File='w15_NoVol_Normalization.json'
#Model_FileName='w15_NoVol_XcryptoAi_model.hdf5'
ALLHIST_FILE='Results_history.json'
#DATA_DIR='/UltimeTradingBot/Data/'
FIRST_NORM_FLAG=True
ORG_DATA_DIR='/UltimeTradingBot/Data'
DATA_DIR=ORG_DATA_DIR+'/'+BUY_MODE
Normalization_File=f'{DATA_DIR}/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Norm_v{VERSION}.json'
Model_FileName=f'{DATA_DIR}/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Model_v{VERSION}.h5'
DATA_FILE=f'{DATA_DIR}/CSV/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Data_v{VERSION}.csv'
REMOTE_DATA_FILE=f'/gdrive/+DATA+/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Data_v{VERSION}.csv.zip'
METAINFO=f"tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min"
window=WINDOW_SIZE
NORM_FILE=Normalization_File
MODEL_FILE=Model_FileName
Px=40
BUFFER_SIZE=250000*Px
SAMPLE_SIZE=5000*Px
ModelTest=f'{DATA_DIR}/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Model_v{VERSION}__TEST__.hdf5'

#DATA_FILE=DATA_DIR+'w'+str(WINDOW_SIZE)+'_EXTData.csv'