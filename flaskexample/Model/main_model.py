import sys,os,math,string,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble, tree, linear_model
from sklearn.linear_model import ElasticNetCV
import seaborn as sns
import scipy.stats as st
import missingno as msno
import pickle
pd.options.display.max_rows = 200


## --- cleaned stored data --
df3 = pd.read_csv("downsized_7.csv")
df3['PTYPE'] = df3['PTYPE'].astype('O') 

## -- all the actual, original features after cleaning all the data 
myfeat = ['key', 'PTYPE','LU', 'OWN_OCC', 'LAND_SF', 'YR_BUILT', 'YR_REMOD', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS','R_BLDG_STYL', 'R_ROOF_TYP', 'R_EXT_FIN', 'R_TOTAL_RMS', 'R_BDRMS','R_FULL_BTH', 'R_HALF_BTH', 'R_BTH_STYLE', 'R_KITCH', 'R_KITCH_STYLE','R_HEAT_TYP', 'R_AC', 'R_FPLACE', 'R_EXT_CND', 'R_OVRALL_CND','R_INT_CND', 'R_INT_FIN', 'R_VIEW', 'R_TOTAL_BTH','MARKET_VALUE', "ZIP_MV",'DIS0', 'DIS1', 'DIS2', 'DIS3', 'DIS4', 'DIS5']
#df3.columns.to_series().groupby(df3.dtypes).groups

## -- features for one-hot-encoding 
dumFeat = ['LU', 'R_BLDG_STYL', 'R_ROOF_TYP','R_EXT_FIN', 'R_BTH_STYLE', 'R_KITCH_STYLE', 'R_HEAT_TYP', 'R_AC','R_EXT_CND', 'R_OVRALL_CND', 'R_INT_CND', 'R_INT_FIN', 'R_VIEW', "PTYPE"]
df3 = pd.get_dummies(df3,columns=dumFeat)

## -- drop market-value from all_features because thats the target feature --
all_features = df3.columns
index = np.argwhere(all_features=="MARKET_VALUE")
all_features = np.delete(all_features, index)
target_feature = "MARKET_VALUE"
#df3.columns.to_series().groupby(df3.dtypes).groups

## -- all the new one-hot-encoded features. I will train the regression model using this feature -- 
my_feat = ['LU_R1', 'LU_R2', 'LU_R3', 'R_BLDG_STYL_BW', 'R_BLDG_STYL_CL', 'R_BLDG_STYL_CN', 'R_BLDG_STYL_CP', 'R_BLDG_STYL_CV', 'R_BLDG_STYL_DK',
        'R_BLDG_STYL_DX', 'R_BLDG_STYL_OT', 'R_BLDG_STYL_RE', 'R_BLDG_STYL_RM', 'R_BLDG_STYL_RN', 'R_BLDG_STYL_RR', 'R_BLDG_STYL_SD', 'R_BLDG_STYL_SL',
        'R_BLDG_STYL_TD', 'R_BLDG_STYL_TF', 'R_BLDG_STYL_TL', 'R_BLDG_STYL_VT', 'R_ROOF_TYP_F', 'R_ROOF_TYP_G', 'R_ROOF_TYP_H', 'R_ROOF_TYP_L',
        'R_ROOF_TYP_M', 'R_ROOF_TYP_S', 'R_EXT_FIN_A', 'R_EXT_FIN_B', 'R_EXT_FIN_C', 'R_EXT_FIN_F', 'R_EXT_FIN_G', 'R_EXT_FIN_M',
        'R_EXT_FIN_O', 'R_EXT_FIN_P', 'R_EXT_FIN_S', 'R_EXT_FIN_U', 'R_EXT_FIN_V', 'R_EXT_FIN_W', 'R_BTH_STYLE_L', 'R_BTH_STYLE_M',
        'R_BTH_STYLE_N', 'R_BTH_STYLE_S', 'R_KITCH_STYLE_L', 'R_KITCH_STYLE_M', 'R_KITCH_STYLE_N', 'R_KITCH_STYLE_S', 'R_HEAT_TYP_E', 'R_HEAT_TYP_F',
        'R_HEAT_TYP_N', 'R_HEAT_TYP_O', 'R_HEAT_TYP_P', 'R_HEAT_TYP_S', 'R_HEAT_TYP_W', 'R_AC_C', 'R_AC_D', 'R_AC_N', 'R_AC_Y', 'R_EXT_CND_A',
        'R_EXT_CND_E', 'R_EXT_CND_F', 'R_EXT_CND_G', 'R_EXT_CND_P', 'R_OVRALL_CND_A', 'R_OVRALL_CND_E', 'R_OVRALL_CND_F', 'R_OVRALL_CND_G',
        'R_OVRALL_CND_P', 'R_INT_CND_A', 'R_INT_CND_E', 'R_INT_CND_F', 'R_INT_CND_G', 'R_INT_CND_P', 'R_INT_FIN_E', 'R_INT_FIN_N',
        'R_INT_FIN_S', 'R_VIEW_A', 'R_VIEW_E', 'R_VIEW_F', 'R_VIEW_G', 'R_VIEW_P', 'PTYPE_101.0', 'PTYPE_104.0', 'PTYPE_105.0',
           'OWN_OCC','LAND_SF', 'YR_BUILT', 'YR_REMOD', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS', 'R_TOTAL_RMS', 'R_BDRMS', 'R_FULL_BTH', 'R_HALF_BTH',
        'R_KITCH', 'R_FPLACE', 'R_TOTAL_BTH', 'DIS0', 'DIS1', 'DIS2', 'DIS3', 'DIS4', 'DIS5', 'ZIP_MV','key']
target_feat = "MARKET_VALUE"
#df3.columns.to_series().groupby(df3.dtypes).groups


## -- test-train split where all 2019 data belongs to test 
## -- [2010 - 2018] data is split into 80-20 train-test split 
test19 = df3[df3["key"]==2019]
train19 = df3[df3["key"]!=2019]
X_train, X_test, Y_train, Y_test = train_test_split(train19[my_feat], train19[target_feat], test_size=0.2, random_state=1) 
X_test = X_test.append(test19[my_feat]);
Y_test = Y_test.append(test19[target_feat]); 


## -- Define and train the model -- 
ElasticNetCVModel2 = ElasticNetCV(l1_ratio=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
                                  eps=0.0001, n_alphas=50, normalize=True, random_state=1,
                                  verbose=False, max_iter=1000,cv=8)
ElasticNetCVModel2.fit(X_train[my_feat],Y_train)
print("train R^2 = ", ElasticNetCVModel2.score(X_train[my_feat],Y_train))
print("test R^2 = ", ElasticNetCVModel2.score(X_test[my_feat],Y_test))
print("L1_ratio = ", ElasticNetCVModel2.l1_ratio_)
#print("mse = ", ElasticNetCVModel2.mse_path_)


## -- save the model into a pickle file 
filename = 'finalized_model.sav'
pickle.dump(ElasticNetCVModel2, open(filename, 'wb'))

## -- print feature and the corresponding weights. 
coef_val = pd.Series(ElasticNetCVModel2.coef_,my_feat)
print(coef_val.sort_values(ascending=False))

## -- plot for visualization
#plt.scatter(X_test["LIVING_AREA"],Y_test)
#plt.scatter(X_test["LIVING_AREA"],ElasticNetCVModel2.predict(X_test[my_feat]))

#plt.scatter(X_test["GROSS_AREA"],Y_test)
#plt.scatter(X_test["GROSS_AREA"],ElasticNetCVModel2.predict(X_test[my_feat]))


## -- residual sum of squares -- 
#df3["Residual"] =  -np.exp(ElasticNetCVModel2.predict(df3[my_feat])) + np.exp(df3["MARKET_VALUE"])
#RSS = (df3["Residual"]**2).sum()/len(df3)
#print(np.sqrt(RSS))




