#!/bin/bash
FILEBASE=/gdrive/TMP/w40.csv
WINDOW=40
WEIGTH=50
PERCENT=0.4
SAMPLES=100000
FORCAST=4

python partial_data_generator.py -w $WINDOW -s 0 -e 10 -o $FILEBASE  -p 1 -S $SAMPLES -W $WEIGTH -P 0.27 -F $FORCAST ;\
python partial_data_generator.py -w $WINDOW -s 10 -e 20 -o $FILEBASE  -p 2 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST &
python partial_data_generator.py -w $WINDOW -s 20 -e 30 -o $FILEBASE  -p 3 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST ;\
python partial_data_generator.py -w $WINDOW -s 30 -e 40 -o $FILEBASE  -p 4 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST &
python partial_data_generator.py -w $WINDOW -s 40 -e 50 -o $FILEBASE  -p 5 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST ;\
python partial_data_generator.py -w $WINDOW -s 50 -e 60 -o $FILEBASE  -p 6 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST &
python partial_data_generator.py -w $WINDOW -s 60 -e 70 -o $FILEBASE  -p 7 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST ;\
python partial_data_generator.py -w $WINDOW -s 70 -e 80 -o $FILEBASE  -p 8 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST &
python partial_data_generator.py -w $WINDOW -s 80 -e 90 -o $FILEBASE  -p 9 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST ;\
python partial_data_generator.py -w $WINDOW -s 90 -e 150 -o $FILEBASE  -p 10 -S $SAMPLES  -W $WEIGTH -P 0.27 -F $FORCAST 