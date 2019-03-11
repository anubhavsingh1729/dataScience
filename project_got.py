import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

got = pd.read_csv("train.csv")
gottest = pd.read_csv("test.csv")

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

labels = got['bestSoldierPerc']
train1 = got.drop(['soldierId', 'shipId','attackId','bestSoldierPerc'],axis=1)

x_train , y_train = train1 , labels

reg.fit(x_train,y_train)

gottest1 =gottest.drop(['Unnamed: 0','index','soldierId', 'shipId','attackId'],axis =1)
x_test,y_test = gottest1,labels
y_test = reg.predict(x_test)
y_test



gottest.insert(26,"bestSoldierPerc",value = y_test)
gottest = gottest.drop([''])

#from sklearn import preprocessing
#import numpy as np
#min_max_scaler = preprocessing.MinMaxScaler()
#y_test1 = pd.DataFrame(y_test)
#x_scaled = min_max_scaler.fit_transform(y_test1)
#x_scaled = np.array(x_scaled)
#gottest.insert(27,"bestSoldierPerc1",value = y_test)
#df_normalized = pd.DataFrame(x_scaled)
#df_normalized

gottest = gottest.drop(['Unnamed: 0'],axis = 1)

gottest.to_csv("gottest.csv")