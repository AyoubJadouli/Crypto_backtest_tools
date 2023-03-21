#!/bin/python
##
WINDOW_SIZE=31
BUY_PCT=0.6
SELL_PCT=0.33
MAX_FORCAST_SIZE=9
VERSION=1
TESTING_MOD=False
UPGRAD_MOD=False
JUST_IMPORT_DATA=False
#Normalization_File='w15_NoVol_Normalization.json'
#Model_FileName='w15_NoVol_XcryptoAi_model.hdf5'
ALLHIST_FILE='Results_history.json'
DATA_DIR='/UltimeTradingBot/Data/'
FIRST_NORM_FLAG=True
DATA_DIR='/UltimeTradingBot/Data'
Normalization_File=f'{DATA_DIR}/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Norm_v{VERSION}.json'
Model_FileName=f'{DATA_DIR}/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Model_v{VERSION}.hdf5'
DATA_FILE=f'{DATA_DIR}/CSV/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Data_v{VERSION}.csv'
REMOTE_DATA_FILE=f'/gdrive/+DATA+/tp{int(BUY_PCT*100)}_w{WINDOW_SIZE}_max{MAX_FORCAST_SIZE}min_Data_v{VERSION}.csv.zip'
window=WINDOW_SIZE
NORM_FILE=Normalization_File
MODEL_FILE=Model_FileName
Px=40
BUFFER_SIZE=250000*Px
SAMPLE_SIZE=5000*Px
DATAPART=""
STARTPOINT=0
ENDPOINT=110
PER_WEIGHT=50

import sys
import getopt

arg_window="" 
arg_start=""
arg_end=""
arg_output=""
arg_part=""
arg_sample=""
arg_weight=""
arg_bp=""
arg_f=""


def argpars(argv):

    global arg_window 
    global arg_start
    global arg_end
    global arg_output
    global arg_part
    global arg_sample
    global arg_weight
    global arg_bp
    global arg_f

    arg_help = "{0} -w <window size> -s <startpoint> -e <endpoint>  -o <output> -p <parts numbre> -S <Sample size> -W <weight pctage> -P <buy signal pct>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "h:w:s:e:o:p:S:W:P:F:", ["help", "window=", 
        "start=", "end=","output=","part=","sample=","weight=","buyp=","forcast="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-w", "--window"):
            arg_window = arg
        elif opt in ("-s", "--start"):
            arg_start = arg
        elif opt in ("-e", "--end"):
            arg_end = arg
        elif opt in ("-o", "--output"):
            arg_output = arg
        elif opt in ("-p", "--part"):
            arg_part = arg
        elif opt in ("-W", "--weight"):
            arg_weight = arg        
        elif opt in ("-S", "--sample"):
            arg_sample = arg
        elif opt in ("-P", "--buyp"):
            arg_bp = arg
        elif opt in ("-F", "--forcast"):
            arg_f = arg

    print('window:', arg_window)
    print('start:', arg_start)
    print('end:', arg_end)
    print('output:', arg_output)
    print('part:', arg_part)
    print('weight:', arg_weight)
    print('sample:', arg_sample)
    print('buyp:', arg_bp)
    print('forcast time:', arg_f)

#if __name__ == "__main__":
argpars(sys.argv)


print('/window:', arg_window)
print('/start:', arg_start)
print('/end:', arg_end)
print('/output:', arg_output)
print('/part:', arg_part)
print('/weight:', arg_weight)
print('/sample:', arg_sample)
print('/buyp:', arg_bp)
print('/forcast time:', arg_f)


#if __name__ == "__main__":
if  arg_window:
    WINDOW_SIZE=int(arg_window)
if  arg_start:
    STARTPOINT=int(arg_start)
if  arg_end:
    ENDPOINT=int(arg_end)
if  arg_output:
    DATA_FILE=arg_output
if  arg_part:
    DATAPART=arg_part
if  arg_sample:
    SAMPLE_SIZE=int(arg_sample)
if  arg_weight:
    PER_WEIGHT=float(arg_weight)
if  arg_bp:
    BUY_PCT=float(arg_bp)
if  arg_f:
    MAX_FORCAST_SIZE=int(arg_f)

DATA_FILE=DATA_FILE+DATAPART

META_INFO=f'Window: {WINDOW_SIZE} - Focast time: {MAX_FORCAST_SIZE}min - Buy treshold: {BUY_PCT}% - Max Down: {SELL_PCT}%'

print(f"working on generataing Data: {META_INFO}")
print(f"file will be saved in {DATA_FILE}")