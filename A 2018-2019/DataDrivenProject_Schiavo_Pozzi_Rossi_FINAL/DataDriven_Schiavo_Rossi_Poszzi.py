# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

ExternalFilesFolder =  r"C:\Users\elbar\Desktop\universita\magistrale\Primo Anno\Building System\DataDrivenProject"
DataFileName= "Project_Data.csv"
path_dataFile = os.path.join(ExternalFilesFolder,DataFileName)
DF_data = pd.read_csv(path_dataFile,sep=";" ,index_col=0)

PreviousIndex = DF_data.index
NewParsedIndex= pd.to_datetime(PreviousIndex)
DF_data.index =NewParsedIndex 

DF_data.columns

def lag_feature(df,column_name,lag_start,lag_end,lag_interval):
    for i in range(lag_start,lag_end,lag_interval):
        new_column_name= column_name + "-" + str(i) + "0min"
        df[new_column_name]=df[column_name].shift(i)
        df=df.dropna()
    return(df)
    
DF_data=DF_data.rename(columns={"T1":"T_kitchen","T2":"T_living_room","T3":"T_laundry","T4":"T_office","T5":"T_bathroom","T6":"T_outside","T7":"T_ironing","T8":"T_teenager","T9":"T_parents"})

DF_data.columns

ListOfTemperature=["T_kitchen","T_living_room","T_laundry","T_office","T_bathroom","T_outside","T_ironing","T_teenager","T_parents"]
for i in ListOfTemperature:
        
    lag_start=1
    lag_end=7
    lag_interval=1
    column_name= i
    df=DF_data
    DF_data=lag_feature(DF_data,column_name,lag_start,lag_end,lag_interval)



DF_data["hour"]=DF_data.index.hour
DF_data["sin_hour"]=np.sin(DF_data.index.hour*2*np.pi/24)
DF_data["cos_hour"]=np.cos(DF_data.index.hour*2*np.pi/24)

for i in ListOfTemperature:
    namecolumn= str(i)+"-24H"
    DF_data[namecolumn]=DF_data[i].shift(+144)

DF_data=DF_data.dropna()

DF_data.head()
DF_data.corr()


DF_data.columns

DF_prediction=DF_data["2016-03-01":"2016-05-15"]

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

ListOfTemperatureToPredict=["T_kitchen","T_living_room","T_laundry","T_office","T_bathroom","T_ironing","T_teenager","T_parents"]

for i in ListOfTemperatureToPredict:
    DF_target=DF_prediction[i]
    DF_features=DF_prediction.drop(i, axis=1)
     
    X_train,X_test,Y_train,Y_test=train_test_split(DF_features,DF_target,test_size=0.2, random_state=41234)    
    
    linear_reg = linear_model.LinearRegression()   
    linear_reg.fit(X_train,Y_train)
    
    predicted_linearReg_split=linear_reg.predict(X_test)    
    predicted_DF_linearReg_split=pd.DataFrame(predicted_linearReg_split,index=Y_test.index, columns=[i+ "_linearReg_split"])
    predicted_DF_linearReg_split=predicted_DF_linearReg_split.join(Y_test)
    
    predicted_DF_linearReg_split_pred=predicted_DF_linearReg_split["2016-04-15":"2016-05-01"]
    predicted_DF_linearReg_split_pred.plot()
    plt.show()
    r2_linearReg_split=r2_score(predicted_linearReg_split,Y_test)
    
    print("The R^2 of "+ i + " is " + str(r2_linearReg_split))
    
    
        
    predict_DF_linearReg_CV = cross_val_predict(linear_reg,DF_features, DF_target, cv=10)
    predicted_DF_linearReg_CV=pd.DataFrame(predict_DF_linearReg_CV,index=DF_target.index, columns=[i+ "inearReg_CV"])
    
    predicted_DF_linearReg_CV=predicted_DF_linearReg_CV.join(DF_target)
    predicted_DF_linearReg_CV_prd=predicted_DF_linearReg_CV["2016-04-15":"2016-05-01"]
    
    predicted_DF_linearReg_CV_prd.plot()
    plt.show()
    r2_linearReg_CV=r2_score(predict_DF_linearReg_CV,DF_target)
    
    print("The R^2 of "+ i + " with cross validation is " + str(r2_linearReg_CV))
    
    
    
    
    reg_RF = RandomForestRegressor()
    predict_RF_CV = cross_val_predict(reg_RF,DF_features, DF_target, cv=10) 
    predicted_DF_RF_CV=pd.DataFrame(predict_RF_CV,index=DF_target.index, columns=[i+ "_RF_CV"])
    predicted_DF_RF_CV=predicted_DF_RF_CV.join(DF_target)    
    
    predicted_DF_RF_CV_pr=predicted_DF_RF_CV["2016-04-15":"2016-05-01"]
    predicted_DF_RF_CV_pr.plot()
    plt.show()
    r2_RF_CV=r2_score(predict_RF_CV,DF_target)
    print("The R^2 of "+ i + " with random forest " + str(r2_RF_CV))
