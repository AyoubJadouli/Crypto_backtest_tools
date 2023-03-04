# Problem
- Need More Memory (128GO or more)
- Need More GPU for traning

# Idee
- Test LSTM with more indicators
- Test Other networks like CNN and Transformer

# Resolution

#Proposed format
-last 10 minutes
-last 3 btc min
-last 200 5min
-last 100 BTC hours
-last 80 hours
-Last 40 days

total: 453*4



# ############## Testing Double good/Bad model ############
## Initial Model
ModelAccuracy: 82.012%
True Win Predictions Mean of all: 40.854%
XXX Loss Buy Mean of all: 8.842%
Missed good deal off all: 9.146%
Good Zero prediction Mean: 41.158%
good fiability
array
## good very deep
ModelAccuracy: 23.433%
True Win Predictions Mean of all: 13.541%
XXX Loss Buy Mean of all: 40.107%
Missed good deal off all: 36.459%
Good Zero prediction Mean: 9.893%
good fiability

## bad very deep
ModelAccuracy: 79.464%
True Win Predictions Mean of all: 38.217%
XXX Loss Buy Mean of all: 8.753%
Missed good deal off all: 11.783%
Good Zero prediction Mean: 41.247%
good fiability

## Retrainedx2
173438/173438 [==============================] - 1000s 6ms/step
ModelAccuracy: 79.066%
True Win Predictions Mean of all: 37.427%
XXX Loss Buy Mean of all: 8.360%
Missed good deal off all: 12.573%
Good Zero prediction Mean: 41.640%
good fiability

## Retrainedx3
#### SNM TEST ######
model_good_x3:
ModelAccuracy: 75.677%
True Win Predictions Mean of all: 10.670%
XXX Loss Buy Mean of all: 15.326%
Missed good deal off all: 8.997%
Good Zero prediction Mean: 65.007%
good fiability
========= Win Ratio:41.04477611940298 ====================
----------------------
very_deep_good_model:
2500/2500 [==============================] - 7s 3ms/step
ModelAccuracy: 74.194%
True Win Predictions Mean of all: 11.447%
XXX Loss Buy Mean of all: 17.586%
Missed good deal off all: 8.219%
Good Zero prediction Mean: 62.747%
check the fiability 99.999
========= Win Ratio:39.427547962663176 ====================
#### ETH TEST ######
2488/2488 [==============================] - 8s 3ms/step
ModelAccuracy: 80.658%
True Win Predictions Mean of all: 5.226%
XXX Loss Buy Mean of all: 14.054%
Missed good deal off all: 5.288%
Good Zero prediction Mean: 75.432%
good fiability
========= Win Ratio:27.105809128630703 ====================



############################################################