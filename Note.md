## Model tp27_w8_max4min val0
- Real time testing: 17/35 
===> Very Bad

## Model tp31_w14_max3min_Model_v1.hdf5val1.h5 Data:10/90
- Testing accuracy full dataset => 98%
- Test NoBuy => 98,6%
- Test Buy => 92,6 %
- Real time testing

## Model tp31_w14_max3min_Model_v1.hdf5val0.h5 Data:10/90
- Testing accuracy full dataset => 98%
- Test NoBuy => 99%
- Test Buy => 89,6 %
- Real time testing



## Model tp31_w14_max3min_Model_v1.hdf5 Data:10/90
- Testing accuracy full dataset => 97.99%
- Test NoBuy => 99.08%
- Test Buy => 90,6 %
- Real time testing

## dessision three model:
- Testing accuracy full dataset => 80.99%
- Test NoBuy => 96.5%
- Test Buy => 30,9 %
- Real time testing


###### relu training #########
- 2500  76.8
-
############## Real Balance ###################
ETC/USDT
2min 0.33% => 0.05507295379609786
b5p => 0.09757070388543627
b9p => 0.05979042676933076
b15p => 0.037082779502681164

===  ETH/USDT  === 
  - 0.33% / 2min => {3.38735921864131}
  - 1.5% / 17min => {1.720005710738923}
  - 0.9% / 11min => {3.2489446777229545}
  - 0.5% / 7min => {6.1582826471502505}
===  DOGE/USDT  === 
  - 0.33% / 2min => {6.202338380790772}
  - 1.5% / 17min => {5.443498580529153}
  - 0.9% / 11min => {7.7699323270817215}
  - 0.5% / 7min => {11.390899187196009}
===  RVN/USDT  === 
  - 0.33% / 2min => {7.236337980550722}
  - 1.5% / 17min => {5.000288489861859}
  - 0.9% / 11min => {7.715534824775307}
  - 0.5% / 7min => {12.324995470203046}
===  LUNA/USDT  === 
  - 0.33% / 2min => {6.929168475481311}
  - 1.5% / 17min => {5.7086742465665665}
  - 0.9% / 11min => {8.40803323579542}
  - 0.5% / 7min => {12.710526693685662}
===  GMT/USDT  === 
  - 0.33% / 2min => {14.763137611415244}
  - 1.5% / 17min => {18.570525109191884}
  - 0.9% / 11min => {23.28504515899524}
  - 0.5% / 7min => {28.63791774540999}
===  APE/USDT  === 
  - 0.33% / 2min => {12.67129024692599}
  - 1.5% / 17min => {12.24131961894551}
  - 0.9% / 11min => {17.132933516489654}
  - 0.5% / 7min => {23.39486938944171}
===  XRP/USDT  === 
  - 0.33% / 2min => {5.297935461660743}
  - 1.5% / 17min => {4.043214564158642}
  - 0.9% / 11min => {6.155645332766319}
  - 0.5% / 7min => {9.595698794082805}


  #full dataset
  buy == 0 - b5 == 1 -> 1.285.457
  buy == 1 - b5 == 0 -> 1.295.651

  buy == 1 - b5 == 1 ->  3.334.901


  ## W7 Np full dataset 7min frocasting 0.5%
  full acc : 79,5%
  buy 1 acc : 30%
  no buy 0 acc: 94,5%







###################### Testing best parameters ###############################
WINDOW_SIZE=1
BUY_PERCENT=0.28
SELL_PERCENT=0.28
MAX_FORCAST_SIZE=7

===  GMT/USDT  === 
  - 0.28% / 7min => {4.729108820884757}
  - 1.5% / 17min => {46.31582067725265}
===  APE/USDT  === 
  - 0.28% / 7min => {4.4971299667210145}
  - 1.5% / 17min => {42.627253031961224}
===  XRP/USDT  === 
  - 0.28% / 7min => {3.0367494064102627}
  - 1.5% / 17min => {22.961215428776825}
===  RVN/USDT  === 
  - 0.28% / 7min => {5.05978289476348}
  - 1.5% / 17min => {29.048150232612752}
===  LUNA/USDT  === 
  - 0.28% / 7min => {3.8780771771380347}
  - 1.5% / 17min => {28.141023800003097}
===  ETH/USDT  === 
  - 0.28% / 7min => {2.5495687603911636}
  - 1.5% / 17min => {18.908626614743756}
===  DOGE/USDT  === 
  - 0.28% / 7min => {4.006795606344861}
  - 1.5% / 17min => {25.432597243891596}



WINDOW_SIZE=1
BUY_PERCENT=0.28
SELL_PERCENT=0.28
MAX_FORCAST_SIZE=5

===  GMT/USDT  === 
  - 0.28% / 5min => {6.30119660484373}
  - 1.5% / 17min => {39.4681975830756}
===  APE/USDT  === 
  - 0.28% / 5min => {5.85955821588904}
  - 1.5% / 17min => {35.7112545507337}
===  XRP/USDT  === 
  - 0.28% / 5min => {3.4187731297493507}
  - 1.5% / 17min => {17.774683967255108}
===  RVN/USDT  === 
  - 0.28% / 5min => {5.6511335063588914}
  - 1.5% / 17min => {22.904561963259738}
===  LUNA/USDT  === 
  - 0.28% / 5min => {4.50174197553502}
  - 1.5% / 17min => {22.308443444891935}
===  ETH/USDT  === 
  - 0.28% / 5min => {2.7249402096762108}
  - 1.5% / 17min => {13.872570163767772}
===  DOGE/USDT  === 
  - 0.28% / 5min => {4.446404109762407}
  - 1.5% / 17min => {20.057871999206235}


  -------
  WINDOW_SIZE=1
BUY_PERCENT=0.28
SELL_PERCENT=0.28
MAX_FORCAST_SIZE=5

===  GMT/USDT  === 
  - 0.28% / 5min ==buynosell=> {35.75190458780867}
  - 0.28% / 5min =====buonly=> {39.4681975830756}
===  APE/USDT  === 
  - 0.28% / 5min ==buynosell=> {33.5112679516673}
  - 0.28% / 5min =====buonly=> {35.7112545507337}
===  XRP/USDT  === 
  - 0.28% / 5min ==buynosell=> {17.21850683704}
  - 0.28% / 5min =====buonly=> {17.774683967255108}
===  RVN/USDT  === 
  - 0.28% / 5min ==buynosell=> {22.250756143284597}
  - 0.28% / 5min =====buonly=> {22.904561963259738}
===  LUNA/USDT  === 
  - 0.28% / 5min ==buynosell=> {21.665889755151365}
  - 0.28% / 5min =====buonly=> {22.308443444891935}
===  ETH/USDT  === 
  - 0.28% / 5min ==buynosell=> {13.71896826505795}
  - 0.28% / 5min =====buonly=> {13.872570163767772}
===  DOGE/USDT  === 
  - 0.28% / 5min ==buynosell=> {19.03083637998293}
  - 0.28% / 5min =====buonly=> {20.057871999206235}


----
WINDOW_SIZE=1
BUY_PERCENT=0.49
SELL_PERCENT=0.49
MAX_FORCAST_SIZE=10

===  GMT/USDT  === 
  - 0.49% / 10min ==buynosell=> {38.33483197138425}
  - 0.49% / 10min =====buonly=> {39.19152594290683}
===  APE/USDT  === 
  - 0.49% / 10min ==buynosell=> {33.47776561767136}
  - 0.49% / 10min =====buonly=> {34.03278761753736}
===  XRP/USDT  === 
  - 0.49% / 10min ==buynosell=> {15.422296698679167}
  - 0.49% / 10min =====buonly=> {15.505222070906763}
===  RVN/USDT  === 
  - 0.49% / 10min ==buynosell=> {19.696064815752074}
  - 0.49% / 10min =====buonly=> {19.819861768504758}
===  LUNA/USDT  === 
  - 0.49% / 10min ==buynosell=> {20.252315822263263}
  - 0.49% / 10min =====buonly=> {20.444297653579945}
===  ETH/USDT  === 
  - 0.49% / 10min ==buynosell=> {11.271159496890498}
  - 0.49% / 10min =====buonly=> {11.293435316083748}
===  DOGE/USDT  === 
  - 0.49% / 10min ==buynosell=> {17.400376836215965}
  - 0.49% / 10min =====buonly=> {17.63172308916601}

---

WINDOW_SIZE=1
BUY_PERCENT=0.99
SELL_PERCENT=0.99
MAX_FORCAST_SIZE=10


===  GMT/USDT  === 
  - 0.99% / 10min ==buynosell=> {20.986532019801786}
  - 0.99% / 10min =====buonly=> {21.123879726885566}
===  APE/USDT  === 
  - 0.99% / 10min ==buynosell=> {14.944274451120096}
  - 0.99% / 10min =====buonly=> {15.009045630178901}
===  XRP/USDT  === 
  - 0.99% / 10min ==buynosell=> {5.196251651672936}
  - 0.99% / 10min =====buonly=> {5.207490621535284}
===  RVN/USDT  === 
  - 0.99% / 10min ==buynosell=> {6.4983783915101405}
  - 0.99% / 10min =====buonly=> {6.510525271583993}
===  LUNA/USDT  === 
  - 0.99% / 10min ==buynosell=> {7.027882654695601}
  - 0.99% / 10min =====buonly=> {7.116804043769206}
===  ETH/USDT  === 
  - 0.99% / 10min ==buynosell=> {2.5816661907741656}
  - 0.99% / 10min =====buonly=> {2.584197533864308}
===  DOGE/USDT  === 
  - 0.99% / 10min ==buynosell=> {6.696588123685705}
  - 0.99% / 10min =====buonly=> {6.735263952406456}





WINDOW_SIZE=1
BUY_PERCENT=0.69
SELL_PERCENT=0.69
MAX_FORCAST_SIZE=10
working on: STMX/USDT -->
===  GMT/USDT  === 
  - 0.69% / 10min ==buynosell=> {29.664140391095124}
  - 0.69% / 10min =====buonly=> {30.02480163631513}
===  APE/USDT  === 
  - 0.69% / 10min ==buynosell=> {23.7620887588502}
  - 0.69% / 10min =====buonly=> {23.99213811895562}
===  XRP/USDT  === 
  - 0.69% / 10min ==buynosell=> {9.45197365423461}
  - 0.69% / 10min =====buonly=> {9.48174173657272}
===  RVN/USDT  === 
  - 0.69% / 10min ==buynosell=> {11.970952760783394}
  - 0.69% / 10min =====buonly=> {12.017819473068341}
===  LUNA/USDT  === 
  - 0.69% / 10min ==buynosell=> {12.75011211827318}
  - 0.69% / 10min =====buonly=> {12.879683285209003}
===  ETH/USDT  === 
  - 0.69% / 10min ==buynosell=> {5.869982118592412}
  - 0.69% / 10min =====buonly=> {5.878284923928077}
===  DOGE/USDT  === 


----
WINDOW_SIZE=1
BUY_PERCENT=0.39
SELL_PERCENT=0.39
MAX_FORCAST_SIZE=10

===  GMT/USDT  === 
  - 0.39% / 10min ==buynosell=> {43.81490667272709}
  - 0.39% / 10min =====buonly=> {45.31881465964448}
===  APE/USDT  === 
  - 0.39% / 10min ==buynosell=> {40.00737051347911}
  - 0.39% / 10min =====buonly=> {40.943202376432225}
===  XRP/USDT  === 
  - 0.39% / 10min ==buynosell=> {20.474163025814192}
  - 0.39% / 10min =====buonly=> {20.64052003017309}
===  RVN/USDT  === 
  - 0.39% / 10min ==buynosell=> {25.932576717670063}
  - 0.39% / 10min =====buonly=> {26.160533167056045}
===  LUNA/USDT  === 
  - 0.39% / 10min ==buynosell=> {25.885402974061467}
  - 0.39% / 10min =====buonly=> {26.1617772789834}
===  ETH/USDT  === 
  - 0.39% / 10min ==buynosell=> {16.22975684930813}
  - 0.39% / 10min =====buonly=> {16.271777144604492}
===  DOGE/USDT  === 
  - 0.39% / 10min ==buynosell=> {22.515407052972726}
  - 0.39% / 10min =====buonly=> {22.902165340180236}


WINDOW_SIZE=1
BUY_PERCENT=0.44
SELL_PERCENT=0.44
MAX_FORCAST_SIZE=10

===  GMT/USDT  === 
  - 0.44% / 10min ==buynosell=> {40.96716500498997}
  - 0.44% / 10min =====buonly=> {42.06298232265842}
===  APE/USDT  === 
  - 0.44% / 10min ==buynosell=> {36.62698501328926}
  - 0.44% / 10min =====buonly=> {37.34616844973534}
===  XRP/USDT  === 
  - 0.44% / 10min ==buynosell=> {17.72324796103824}
  - 0.44% / 10min =====buonly=> {17.83796645521878}
===  RVN/USDT  === 
  - 0.44% / 10min ==buynosell=> {22.53894087303676}
  - 0.44% / 10min =====buonly=> {22.70656781805593}
===  LUNA/USDT  === 
  - 0.44% / 10min ==buynosell=> {22.852907453269324}
  - 0.44% / 10min =====buonly=> {23.076923076923077}
===  ETH/USDT  === 
  - 0.44% / 10min ==buynosell=> {13.473326731590554}
  - 0.44% / 10min =====buonly=> {13.50319658005423}
===  DOGE/USDT  === 
  - 0.44% / 10min ==buynosell=> {19.729026209454926}
  - 0.44% / 10min =====buonly=> {20.02770080297905}


---

===  GMT/USDT  === 
  - 0.4% / 10min ==buynosell=> {43.22105075936484}
  - 0.4% / 10min =====buonly=> {44.64096914123099}
===  APE/USDT  === 
  - 0.4% / 10min ==buynosell=> {39.304938244031}
  - 0.4% / 10min =====buonly=> {40.187166372590625}
===  XRP/USDT  === 
  - 0.4% / 10min ==buynosell=> {19.875257559726013}
  - 0.4% / 10min =====buonly=> {20.029363074415144}
===  RVN/USDT  === 
  - 0.4% / 10min ==buynosell=> {25.1691453050284}
  - 0.4% / 10min =====buonly=> {25.383638962332526}
===  LUNA/USDT  === 
  - 0.4% / 10min ==buynosell=> {25.242186516203574}
  - 0.4% / 10min =====buonly=> {25.50718330177821}
===  ETH/USDT  === 
  - 0.4% / 10min ==buynosell=> {15.62061444809633}
  - 0.4% / 10min =====buonly=> {15.660103400302546}
===  DOGE/USDT  === 
  - 0.4% / 10min ==buynosell=> {21.936889552160228}
  - 0.4% / 10min =====buonly=> {22.30420867938244}











WINDOW_SIZE=1
BUY_PERCENT=0.44
SELL_PERCENT=0.44
MAX_FORCAST_SIZE=10

  ===  GMT/USDT  === 
  - 0.43% / 10min ==buynosell=> {41.54125865834017}
  - 0.43% / 10min =====buonly=> {42.705255773050204}
===  APE/USDT  === 
  - 0.43% / 10min ==buynosell=> {37.26017912581243}
  - 0.43% / 10min =====buonly=> {38.022915596453224}
===  XRP/USDT  === 
  - 0.43% / 10min ==buynosell=> {18.2270778172098}
  - 0.43% / 10min =====buonly=> {18.34939020994598}
===  RVN/USDT  === 
  - 0.43% / 10min ==buynosell=> {23.182118172947277}
  - 0.43% / 10min =====buonly=> {23.36067731003292}
===  LUNA/USDT  === 
  - 0.43% / 10min ==buynosell=> {23.40841759693536}
  - 0.43% / 10min =====buonly=> {23.641049206114257}
===  ETH/USDT  === 
  - 0.43% / 10min ==buynosell=> {13.980000364513407}
  - 0.43% / 10min =====buonly=> {14.011186511383954}
===  DOGE/USDT  === 
  - 0.43% / 10min ==buynosell=> {20.2378867203201}
  - 0.43% / 10min =====buonly=> {20.55053321008366}

  WINDOW_SIZE=1
BUY_PERCENT=0.44
SELL_PERCENT=0.44
MAX_FORCAST_SIZE=14


===  GMT/USDT  === 
  - 0.44% / 14min ==buynosell=> {47.92545675523453}
  - 0.44% / 14min =====buonly=> {48.88787881782161}
===  APE/USDT  === 
  - 0.44% / 14min ==buynosell=> {43.613338395908244}
  - 0.44% / 14min =====buonly=> {44.24541576396489}
===  XRP/USDT  === 
  - 0.44% / 14min ==buynosell=> {22.799009755628344}
  - 0.44% / 14min =====buonly=> {22.8961104051598}
===  RVN/USDT  === 
  - 0.44% / 14min ==buynosell=> {28.61987148600882}
  - 0.44% / 14min =====buonly=> {28.769480558918442}
===  LUNA/USDT  === 
  - 0.44% / 14min ==buynosell=> {28.80114084485261}
  - 0.44% / 14min =====buonly=> {29.01300571968885}
===  ETH/USDT  === 
  - 0.44% / 14min ==buynosell=> {18.351933642359697}
  - 0.44% / 14min =====buonly=> {18.375728267407034}
===  DOGE/USDT  === 
  - 0.44% / 14min ==buynosell=> {24.935177488642772}
  - 0.44% / 14min =====buonly=> {25.19760614844431}


====

WINDOW_SIZE=1
BUY_PERCENT=0.44
SELL_PERCENT=0.44
MAX_FORCAST_SIZE=11

  ===  GMT/USDT  === 
  - 0.44% / 11min ==buynosell=> {42.97797496121657}
  - 0.44% / 11min =====buonly=> {44.04019643686452}
===  APE/USDT  === 
  - 0.44% / 11min ==buynosell=> {38.60362271904943}
  - 0.44% / 11min =====buonly=> {39.29823777723181}
===  XRP/USDT  === 
  - 0.44% / 11min ==buynosell=> {19.085492109939402}
  - 0.44% / 11min =====buonly=> {19.194337989236914}
===  RVN/USDT  === 
  - 0.44% / 11min ==buynosell=> {24.185551691048154}
  - 0.44% / 11min =====buonly=> {24.350546812051324}
===  LUNA/USDT  === 
  - 0.44% / 11min ==buynosell=> {24.492153930104475}
  - 0.44% / 11min =====buonly=> {24.71296617452452}
===  ETH/USDT  === 
  - 0.44% / 11min ==buynosell=> {14.764311707563046}
  - 0.44% / 11min =====buonly=> {14.790941436871341}
===  DOGE/USDT  === 
  - 0.44% / 11min ==buynosell=> {21.13563167651618}
  - 0.44% / 11min =====buonly=> {21.421549321299953}












  ### test training with differentt windows

  ## w2
  ------val_accuracy-----> 68.34 | 70.16 <----------accuracy----------

## w35
mini relu acc: ------val_accuracy-----> 70.94 | 81.64 <----------accuracy----------
softplus valacc: ------val_accuracy-----> 64.2 | 63.31 <----------accuracy----------
softplus acc: ------val_accuracy-----> 76.33 | 89 <----------accuracy----------
        b==1 >>> loss: 0.7342 - accuracy: 0.7889 
        b==0 >>> loss: 0.6519 - accuracy: 0.7367


## w15
mini relu acc: ------val_accuracy-----> 70.71 | 76.7 <----------accuracy----------
softplus valacc: 
softplus acc: ------val_accuracy-----> 73.85 | 82.46 <----------accuracy----------
        b==1 >>>  loss: 0.6418 - accuracy: 0.7827
        b==0 >>> loss: 0.5455 - accuracy: 0.6928

## w8
mini relu acc:  ------val_accuracy-----> 69.92 | 74.63 <----------accuracy----------
softplus valacc: 
softplus acc: ------val_accuracy-----> 73.85 | 82.46 <----------accuracy----------
        b==1 >>>  loss: 0.6418 - accuracy: 0.7827
        b==0 >>> loss: 0.5455 - accuracy: 0.6928


## w18
maxi relu acc:  ------val_accuracy-----> 76.74 | 98.92 <----------accuracy----------
        b==1 >>>  loss: 2.5030 - accuracy: 0.7526
        b==0 >>> loss: 1.6334 - accuracy: 0.7773

## max w6
maxi relu acc: ------val_accuracy-----> 79.34 | 79.94 <----------accuracy----------
        b==1 >>>  104s 2ms/step - loss: 0.8061 - accuracy: 0.5284
        b==0 >>> 90%