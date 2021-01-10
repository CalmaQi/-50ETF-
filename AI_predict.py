import  pandas_datareader
from datetime import date

code='510050.SS'
stock=pandas_datareader.get_data_yahoo(code,'2010-01-01',(date.today()))
stock.to_csv('.//data//'+code+'.csv')
# stock.to_csv('.//wjq_data//'+code+'.csv')
print("数据更新到:")
print(stock.tail(1))


import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm,preprocessing
import matplotlib.pyplot as plt
import talib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

origDf=pd.read_csv('.\\data\\510050.SS.csv',encoding='gbk')
df=origDf[['Date','Close', 'High','Low','Open' ,'Volume']]
# 调用talib计算指数移动平均线的值
close = df['Close'].values  
open=df['Open'].values
high=df['High'].values
low=df['Low'].values
volume=df['Volume'].values
df['EMA12'] = talib.EMA(np.array(close), timeperiod=6)  
df['EMA26'] = talib.EMA(np.array(close), timeperiod=12)   
df['SAR']=talib.SAR(high, low, acceleration=0, maximum=0)
df['SMA']=talib.SMA(close, timeperiod=30)
df['MACD'],df['MACDsignal'],df['MACDhist'] = talib.MACD(np.array(close),
                            fastperiod=6, slowperiod=12, signalperiod=9) 
df['RSI']=talib.RSI(np.array(close), timeperiod=12)     #RSI的天数一般是6、12、24
df['MOM']=talib.MOM(np.array(close), timeperiod=5)
df['CDL2CROWS'] =talib.CDL2CROWS(open, high, low, close)
df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open, high, low, close)
df['CDL3INSIDE'] = talib.CDL3INSIDE(open, high, low, close)
df['CDL3LINESTRIKE']= talib.CDL3LINESTRIKE(open, high, low, close)
df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(open, high, low, close)
df['CDLADVANCEBLOCK']=talib.CDLADVANCEBLOCK(open, high, low, close)
df['BETA']=talib.BETA(high, low, timeperiod=5)
df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
df['AD'] = talib.AD(high, low, close, volume)
df['TSF']=talib.TSF(close, timeperiod=14)
df['CDLDARKCLOUDCOVER']=talib.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
df['CDLDOJI']=talib.CDLDOJI(open, high, low, close)
df['CDLMORNINGDOJISTAR']=talib.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
df['CDLMORNINGSTAR']=talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
df['CDLONNECK']=talib.CDLONNECK(open, high, low, close)
df['CDLPIERCING']=talib.CDLPIERCING(open, high, low, close)
df['CDLRICKSHAWMAN']=talib.CDLRICKSHAWMAN(open, high, low, close)
df['CDLRISEFALL3METHODS']=talib.CDLRISEFALL3METHODS(open, high, low, close)
df['CDLSEPARATINGLINES']=talib.CDLSEPARATINGLINES(open, high, low, close)
df['CDLSHOOTINGSTAR']=talib.CDLSHOOTINGSTAR(open, high, low, close)
df['CDLSHORTLINE']=talib.CDLSHORTLINE(open, high, low, close)
df['CDLSPINNINGTOP']=talib.CDLSPINNINGTOP(open, high, low, close)
df['CDLSTALLEDPATTERN']=talib.CDLSTALLEDPATTERN(open, high, low, close)
df['CDLUPSIDEGAP2CROWS']=talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
df['CDLXSIDEGAP3METHODS']=talib.CDLXSIDEGAP3METHODS(open, high, low, close)
df['CDLUNIQUE3RIVER']=talib.CDLUNIQUE3RIVER(open, high, low, close)
df['ADOSC']=talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
df['OBV']=talib.OBV(close, volume)
df['CCI']=talib.CCI(high, low, close, timeperiod=14)
df['ROC']=talib.ROC(close, timeperiod=10)
df['ROCR']=talib.ROCR(close, timeperiod=10)
df['HT_DCPERIOD']=talib.HT_DCPERIOD(close)
df['ADXR']=talib.ADXR(np.array(high),np.array(low),np.array(close),timeperiod=14)
df['BOP']=talib.BOP(np.array(open),np.array(high),np.array(low),np.array(close))
df['CMO']=talib.CMO(np.array(close),timeperiod=14)
df ['DX']=talib.DX(np.array(high),np.array(low),np.array(close),timeperiod=14) #21
df ['MFI']=talib.MFI(np.array(high),np.array(low),np.array(close),np.array(volume),timeperiod=14)
df['MINUS_DI']=talib.MINUS_DI(np.array(high),np.array(low),np.array(close),timeperiod=14)
df ['NATR']=talib.NATR(np.array(high),np.array(low),np.array(close),timeperiod=14)
df ['PPO']=talib.PPO(np.array(close))
df ['T3']=talib.T3(np.array(close),timeperiod=5,vfactor=0.7) #30
df['TRIX']=talib.TRIX(np.array(close),timeperiod=30)
df['ULT']=talib.ULTOSC(np.array(high),np.array(low),np.array(close),timeperiod1=7,timeperiod2=14,timeperiod3=28)
df['upperband'],df['middleband'],df['lowerband']=talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
#diff列表示本日和明日收盘价的差
df['diff'] = df["Close"].shift(-1)-df['Close']
df['diff'].fillna(0, inplace = True)
#up列表示本日是否上涨,1表示涨，0表示跌
df['up'] = df['diff']   
df['up'][df['diff']>0] = 1
df['up'][df['diff']<=0] = 0
#预测值暂且初始化为0
df['predictForUp'] = 0
df.fillna(0, inplace = True)


##SVM
target = df['up']
length=len(df)
trainNum=int(length-3)
predictNum=length-trainNum
#选择指定列作为特征列'
feature=df[['Close', 'High', 'Low','Open' ,'Volume','MACD','EMA12','CDLBREAKAWAY','CDLADVANCEBLOCK','ADXR','CDLDARKCLOUDCOVER'
,'CDLDOJI','CDLMORNINGDOJISTAR','CDLMORNINGSTAR','CDLONNECK','CDLPIERCING','CDLRICKSHAWMAN','CDLRISEFALL3METHODS','CDLSEPARATINGLINES'
,'CDLUPSIDEGAP2CROWS','CDLUNIQUE3RIVER']]
#标准化处理特征值
# feature=preprocessing.scale(feature)
stand_means = preprocessing.StandardScaler()
feature = stand_means.fit_transform(feature)
# Y_trans = stand_means.transform(feature.loc[4:8,:])
# TT=stand_means.transform(feature.loc[4:6,:])
#训练集的特征值和目标值
featureTrain=feature[0:trainNum]
targetTrain=target[0:trainNum]
featureTest=feature[trainNum:length]
targetTest=target[trainNum:length]
#目标值是真实的涨跌情况
svmTool = svm.SVC(kernel='rbf')
svmTool.fit(featureTrain,targetTrain)
svm_predict=svmTool.predict(featureTest)
#预测完毕


length=len(df)
trainNum=int(length-3)
feature=df.drop(['Date','diff','up','predictForUp'],1)
#标准化处理特征值
# print('选取特征前的特征数量:',feature.columns.shape)
stand_means = preprocessing.StandardScaler()
feature = stand_means.fit_transform(feature)
#划分训练集和测试机
featureTrain=feature[0:trainNum]
targetTrain=target[0:trainNum]
featureTest=feature[trainNum:length]
targetTest=target[trainNum:length]
logistic=LogisticRegression()
logistic.fit(featureTrain,targetTrain)
model = SelectFromModel(logistic, prefit=True)
feature=model.transform(feature)
featureTrain=feature[0:trainNum]
featureTest=feature[trainNum:length]
logistic.fit(featureTrain,targetTrain)
#logistics结果
logistic_predict=logistic.predict(featureTest)


#目标值是真实的涨跌情况
target = df['up']
length=len(df)
trainNum=int(length-1)
#选择指定列作为特征列'
feature=df[['Close', 'High', 'Low', 'Open', 'Volume', 'EMA12', 'EMA26',
            'SMA', 'MACD', 'MACDsignal', 'MACDhist', 'RSI', 'MOM', 'BETA',
            'AD', 'ADOSC', 'CCI', 'ROCR', 'ADXR', 'CMO', 'MFI', 'NATR', 'T3',
            'ULT', 'middleband']]

#训练集的特征值和目标值
featureTrain=feature[0:trainNum]
targetTrain=target[0:trainNum]
featureTest=feature[trainNum:length]
targetTest=target[trainNum:length]


rfc = RandomForestClassifier(max_features=4,random_state=0)
rfc.fit(featureTrain, targetTrain)
#rfc结果
rfc_predict= rfc.predict(featureTest)



feature=df[['Close', 'High', 'Low', 'Open', 'Volume', 'EMA12', 'EMA26', 'SMA',
       'MACD', 'MACDhist', 'RSI', 'MOM', 'BETA', 'CORREL', 'AD', 'TSF',
       'CDLDOJI', 'CDLRICKSHAWMAN', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'ADOSC',
       'OBV', 'CCI', 'ROC', 'ROCR', 'HT_DCPERIOD', 'ADXR', 'BOP', 'CMO', 'DX',
       'MFI', 'MINUS_DI', 'NATR', 'PPO', 'T3', 'TRIX', 'ULT', 'upperband',
       'middleband', 'lowerband']]
length=len(feature)
target = df['up']
trainNum=int(length-1)
featureTrain=feature[0:trainNum]
targetTrain=target[0:trainNum]
featureTest=feature[trainNum:length]
targetTest=target[trainNum:length]
xgbTool = xgb.XGBClassifier(objective='binary:logistic',max_depth=6,min_child_weight= 1,n_estimators=100)
xgbTool.fit(featureTrain, targetTrain,verbose=True)
# 进行预测xgbost结果
xgb_predict = xgbTool.predict(featureTest)
xgb_predict = [round(value) for value in xgb_predict]

print("svm预测结果",svm_predict[-1])
print("逻辑回归预测结果",logistic_predict[-1])
print("随机森林预测结果",rfc_predict[-1])
print("xgboost预测结果",xgb_predict[-1])
final_result=0.2*svm_predict[-1]+0.2*logistic_predict[-1]+0.3*rfc_predict[-1]+0.3*xgb_predict[-1]
if(final_result>=0.5):
    print("最终预测结果是:涨")
else:
    print("最终预测结果是:跌")


