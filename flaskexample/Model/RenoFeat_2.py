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
import seaborn as sns
import scipy.stats as st
import missingno as msno
import pickle
pd.options.display.max_rows = 200
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings(action='once')


## -- Collect profits of all the renovations to suggest one with maximum profit -- 
def main(total, cmdargs):
	if total != 1:
		print (" ".join(str(x) for x in cmdargs))
		raise ValueError('I did not ask for arguments')
	#os.system("pwd")
	df3 = pd.read_csv("downsized_7.csv")
	df3['PTYPE'] = df3['PTYPE'].astype('O') 

	## --- get the features for one-hot-encoding
	dumFeat = ['LU', 'R_BLDG_STYL', 'R_ROOF_TYP','R_EXT_FIN', 'R_BTH_STYLE', 'R_KITCH_STYLE', 'R_HEAT_TYP', 'R_AC','R_EXT_CND', 'R_OVRALL_CND', 'R_INT_CND', 'R_INT_FIN', 'R_VIEW', "PTYPE"]
	df3 = pd.get_dummies(df3,columns=dumFeat)

	## ---- collecting all features
	all_features = df3.columns
	index = np.argwhere(all_features=="MARKET_VALUE")
	all_features = np.delete(all_features, index)
	target_feature = "MARKET_VALUE"

	my_feat = ['LU_R1', 'LU_R2', 'LU_R3', 'R_BLDG_STYL_BW', 'R_BLDG_STYL_CL',
		'R_BLDG_STYL_CN', 'R_BLDG_STYL_CP', 'R_BLDG_STYL_CV', 'R_BLDG_STYL_DK',
		'R_BLDG_STYL_DX', 'R_BLDG_STYL_OT', 'R_BLDG_STYL_RE', 'R_BLDG_STYL_RM',
		'R_BLDG_STYL_RN', 'R_BLDG_STYL_RR', 'R_BLDG_STYL_SD', 'R_BLDG_STYL_SL',
		'R_BLDG_STYL_TD', 'R_BLDG_STYL_TF', 'R_BLDG_STYL_TL', 'R_BLDG_STYL_VT',
		'R_ROOF_TYP_F', 'R_ROOF_TYP_G', 'R_ROOF_TYP_H', 'R_ROOF_TYP_L',
		'R_ROOF_TYP_M', 'R_ROOF_TYP_S', 'R_EXT_FIN_A', 'R_EXT_FIN_B',
		'R_EXT_FIN_C', 'R_EXT_FIN_F', 'R_EXT_FIN_G', 'R_EXT_FIN_M',
		'R_EXT_FIN_O', 'R_EXT_FIN_P', 'R_EXT_FIN_S', 'R_EXT_FIN_U',
		'R_EXT_FIN_V', 'R_EXT_FIN_W', 'R_BTH_STYLE_L', 'R_BTH_STYLE_M',
		'R_BTH_STYLE_N', 'R_BTH_STYLE_S', 'R_KITCH_STYLE_L', 'R_KITCH_STYLE_M',
		'R_KITCH_STYLE_N', 'R_KITCH_STYLE_S', 'R_HEAT_TYP_E', 'R_HEAT_TYP_F',
		'R_HEAT_TYP_N', 'R_HEAT_TYP_O', 'R_HEAT_TYP_P', 'R_HEAT_TYP_S',
		'R_HEAT_TYP_W', 'R_AC_C', 'R_AC_D', 'R_AC_N', 'R_AC_Y', 'R_EXT_CND_A',
		'R_EXT_CND_E', 'R_EXT_CND_F', 'R_EXT_CND_G', 'R_EXT_CND_P',
		'R_OVRALL_CND_A', 'R_OVRALL_CND_E', 'R_OVRALL_CND_F', 'R_OVRALL_CND_G',
		'R_OVRALL_CND_P', 'R_INT_CND_A', 'R_INT_CND_E', 'R_INT_CND_F',
		'R_INT_CND_G', 'R_INT_CND_P', 'R_INT_FIN_E', 'R_INT_FIN_N',
		'R_INT_FIN_S', 'R_VIEW_A', 'R_VIEW_E', 'R_VIEW_F', 'R_VIEW_G',
		'R_VIEW_P', 'PTYPE_101.0', 'PTYPE_104.0', 'PTYPE_105.0',
		   'OWN_OCC','LAND_SF', 'YR_BUILT', 'YR_REMOD', 'GROSS_AREA', 'LIVING_AREA',
		'NUM_FLOORS', 'R_TOTAL_RMS', 'R_BDRMS', 'R_FULL_BTH', 'R_HALF_BTH',
		'R_KITCH', 'R_FPLACE', 'R_TOTAL_BTH', 'DIS0', 'DIS1',
		'DIS2', 'DIS3', 'DIS4', 'DIS5', 'ZIP_MV','key']
	target_feat = "MARKET_VALUE"

	filemodel = open('finalized_model.sav','rb')
	ElasticNetCVModel2 = pickle.load(filemodel)
	filemodel.close(); 

	df3["Residual"] =  -np.exp(ElasticNetCVModel2.predict(df3[my_feat])) + np.exp(df3["MARKET_VALUE"])

	## prediction - marketvalue ---> (negative is under-valued!)
	predic_y = np.exp(ElasticNetCVModel2.predict(df3[my_feat]))
	df3["Residual"] =  predic_y - np.exp(df3["MARKET_VALUE"])

	recommend = df3[(df3["key"]==2019)]
	recommend["MARKET_VALUE"] = np.exp(recommend["MARKET_VALUE"]); 
	recommend["Appreciation_2020"] = Appreciation(recommend,ElasticNetCVModel2,my_feat)
	recommend["1yr_Increase"] = recommend["Appreciation_2020"] - recommend["MARKET_VALUE"]
	recommend = recommend[(recommend["Residual"]<15000) & (recommend["Residual"]>-15000)]

	recommend["MVReno19_INT_COND"], recommend["MVReno20_INT_COND"], recommend["Prof19_INT_COND"], recommend["Prof20_INT_COND"], recommend["Expected_RenoCost_INT_COND"] = Renovate(recommend,ElasticNetCVModel2,'Interior Condition',my_feat); 

	feature_of_interest="Exterior Condition"
	recommend["MVReno19_EXT_COND"], recommend["MVReno20_EXT_COND"], recommend["Prof19_EXT_COND"], recommend["Prof20_EXT_COND"], recommend["Expected_RenoCost_EXT_COND"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Interior Finish"
	recommend["MVReno19_INT_FIN"], recommend["MVReno20_INT_FIN"], recommend["Prof19_INT_FIN"], recommend["Prof20_INT_FIN"], recommend["Expected_RenoCost_INT_FIN"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Exterior Finish"
	recommend["MVReno19_EXT_FIN"], recommend["MVReno20_EXT_FIN"], recommend["Prof19_EXT_FIN"], recommend["Prof20_EXT_FIN"], recommend["Expected_RenoCost_EXT_FIN"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Fireplace"
	recommend["MVReno19_FRPL"], recommend["MVReno20_FRPL"], recommend["Prof19_FRPL"], recommend["Prof20_FRPL"], recommend["Expected_RenoCost_FRPL"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Mansard Roof"
	recommend["MVReno19_ROOF_M"], recommend["MVReno20_ROOF_M"], recommend["Prof19_ROOF_M"], recommend["Prof20_ROOF_M"], recommend["Expected_RenoCost_ROOF_M"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Luxury Kitchen"
	recommend["MVReno19_KITCHEN_L"], recommend["MVReno20_KITCHEN_L"], recommend["Prof19_KITCHEN_L"], recommend["Prof20_KITCHEN_L"], recommend["Expected_RenoCost_KITCHEN_L"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	feature_of_interest="Modern Kitchen"
	recommend["MVReno19_KITCHEN_M"], recommend["MVReno20_KITCHEN_M"], recommend["Prof19_KITCHEN_M"], recommend["Prof20_KITCHEN_M"], recommend["Expected_RenoCost_KITCHEN_M"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 

	recommend['Max_Prof'] = recommend.apply(lambda row: np.max([row['Prof20_INT_COND'],row['Prof20_EXT_COND'],row['Prof20_INT_FIN'],row['Prof20_EXT_FIN'],row['Prof20_FRPL'],row['Prof20_ROOF_M'],row['Prof20_KITCHEN_L'],row['Prof20_KITCHEN_M']]), axis=1)

	recommend['PER_Max_Prof'] = recommend["Max_Prof"]/recommend["MARKET_VALUE"]
	recommend = recommend[(recommend["PER_Max_Prof"]<0.5) & (recommend["PER_Max_Prof"]>-0.2)]
	recommend = recommend.sort_values(by="Max_Prof",ascending=False)

	originalData = pd.read_csv("pas2019.csv")
	zillow = pd.read_csv("Add2_LatLong.csv"); 
	zillow1 = pd.read_csv("Add2_LatLong_2.csv"); 
	zillow2 = pd.read_csv("Add2_LatLong_3.csv"); 
	zillow = zillow.append(zillow1); 
	zillow = zillow.append(zillow2); 
	originalData["ST_NUM"] = originalData["ST_NUM"].astype(str); 
	originalData["ST_NAME_SUF"] = originalData["ST_NAME_SUF"].astype(str); 
	originalData["UNIT_NUM"] = originalData["UNIT_NUM"].astype(str); 
	originalData["ZIPCODE"] = originalData["ZIPCODE"].astype(str); 
	originalData["ZIPCODE"] = originalData["ZIPCODE"].map(addressfix); 


	recommend["latitude"] = 0.0; 
	recommend["longitude"] = 0.0; 
	for index, row in recommend.iterrows(): 
		vid = row["PID"]
		temper = originalData[originalData["PID"]==vid]
		string = temper.iloc[0]["ST_NUM"] + " " + temper.iloc[0]["ST_NAME"].title() + " "  + temper.iloc[0]["ST_NAME_SUF"].title() + " Boston, MA " + temper.iloc[0]["ZIPCODE"]
		recommend.set_value(index,"FULLADD",string)
		#print(row["latitude"])


		temper = zillow[zillow["PID"]==vid]
		try: 
			latitude = temper.iloc[0]["latitude"]
			longitude = temper.iloc[0]["longitude"]
			recommend.set_value(index,"latitude",latitude)
			recommend.set_value(index,"longitude",longitude)
		except: 
			pass; 

	#print(recommend["latitude"])
	#interFeat = ["PID","FULLADD","MARKET_VALUE", "MVReno19","MVReno20",'Prof19','Prof20']
	recommend.to_csv("AllReno.csv",index=False); 
	#return(recommend);



def Renovate(dataf,model,feature_of_interest,my_feat):
	
	rec = dataf.copy(deep=True); 
	## -- Renovate interior condition to excellent
	if feature_of_interest == 'Interior Condition':
		#R_INT_CND_A       -0.005797
		#R_INT_CND_E        0.106168
		#R_INT_CND_F       -0.030033
		#R_INT_CND_G        0.038299
		#R_INT_CND_P       -0.004748
		#rec = rec[rec["R_INT_CND_E"]!=1]
		BuildStylFeat = ['R_INT_CND_A', 'R_INT_CND_E', 'R_INT_CND_F','R_INT_CND_G', 'R_INT_CND_P']
		rec[BuildStylFeat] = 0;
		rec["R_INT_CND_E"] = 1;
		rec["Expected_RenoCost"] = 3000*rec["NUM_FLOORS"]; 

	## -- Renovate exterior condition
	elif feature_of_interest == 'Exterior Condition':
		#R_EXT_CND_A       -0.000000
		#R_EXT_CND_E       -0.048922
		#R_EXT_CND_F       -0.009762
		#R_EXT_CND_G        0.011999
		#R_EXT_CND_P       -0.143379
		#rec = rec[rec["R_EXT_CND_G"]!=1]
		#red = rec[rec["R_EXT_CND_E"]!=1]
		BuildStylFeat = ['R_EXT_CND_A','R_EXT_CND_E', 'R_EXT_CND_F', 'R_EXT_CND_G', 'R_EXT_CND_P']
		rec[BuildStylFeat] = 0;
		rec["R_EXT_CND_G"] = 1;
		rec["Expected_RenoCost"] = 1500*rec["NUM_FLOORS"] + 300; 

	# -- Renovate Interior Finish 
	elif feature_of_interest == 'Interior Finish':
		#R_INT_FIN_E        0.093323	Elaborate
		#R_INT_FIN_N       -0.000000	Normal
		#R_INT_FIN_S       -0.084358	substandard
		#rec = rec[rec["R_INT_FIN_E"]!=1]
		BuildStylFeat = ['R_INT_FIN_E', 'R_INT_FIN_N','R_INT_FIN_S']
		rec[BuildStylFeat] = 0;
		rec["R_INT_FIN_E"] = 1;
		rec["Expected_RenoCost"] = 4000*rec["NUM_FLOORS"];

	## -- Renovate Exterior Finish
	elif feature_of_interest == 'Exterior Finish':
		#R_EXT_FIN_A       -0.083155
		#R_EXT_FIN_B        0.000000
		#R_EXT_FIN_C        0.041162
		#R_EXT_FIN_F       -0.000000
		#R_EXT_FIN_G        0.171578
		#R_EXT_FIN_M       -0.021715
		#R_EXT_FIN_O       -0.000000
		#R_EXT_FIN_P       -0.108465
		#R_EXT_FIN_S        0.015849
		#R_EXT_FIN_U       -0.028644
		#R_EXT_FIN_V        0.023906
		#R_EXT_FIN_W        0.016915
		#rec = rec[rec["R_EXT_FIN_B"]!=1]
		BuildStylFeat = ['R_EXT_FIN_A', 'R_EXT_FIN_B','R_EXT_FIN_C', 'R_EXT_FIN_F', 'R_EXT_FIN_G', 'R_EXT_FIN_M','R_EXT_FIN_O', 'R_EXT_FIN_P', 'R_EXT_FIN_S', 'R_EXT_FIN_U','R_EXT_FIN_V', 'R_EXT_FIN_W']
		rec[BuildStylFeat] = 0;
		rec["R_EXT_FIN_B"] = 1;
		rec["Expected_RenoCost"] = 5000*rec["NUM_FLOORS"];

	## -- add a Fireplace
	elif feature_of_interest == 'Fireplace':
		#R_FPLACE           0.015061
		#rec = rec[rec["R_FPLACE"]==0]
		rec["R_FPLACE"] = 1;
		rec["Expected_RenoCost"] = 3200

	## Renovate to Mansard Roof
	elif feature_of_interest == 'Mansard Roof':
		#R_ROOF_TYP_M       0.043549
		#R_ROOF_TYP_H       0.010833
		#R_ROOF_TYP_G       0.000000
		#R_ROOF_TYP_F      -0.002714
		#R_ROOF_TYP_L      -0.006662
		#R_ROOF_TYP_S      -0.026214
		#rec = rec[rec["R_ROOF_TYP_M"]!=1]
		BuildStylFeat = ['R_ROOF_TYP_F', 'R_ROOF_TYP_G', 'R_ROOF_TYP_H', 'R_ROOF_TYP_L','R_ROOF_TYP_M', 'R_ROOF_TYP_S']
		rec[BuildStylFeat] = 0;
		rec["R_ROOF_TYP_M"] = 1;
		rec["Expected_RenoCost"] = 16000; 

	## Renovate to Luxury Kitchen
	elif feature_of_interest == 'Luxury Kitchen':
		#R_KITCH_STYLE_L    0.012301 - luxury
		#R_KITCH_STYLE_M    0.022410 - Modern
		#R_KITCH_STYLE_S   -0.000000 - Semi-Modern
		#R_KITCH_STYLE_N   -0.027267 - No Remodeling
		#rec = rec[rec["R_KITCH_STYLE_L"]!=1]
		BuildStylFeat = ['R_KITCH_STYLE_L', 'R_KITCH_STYLE_M','R_KITCH_STYLE_N', 'R_KITCH_STYLE_S']
		rec[BuildStylFeat] = 0;
		rec["R_KITCH_STYLE_L"] = 1;
		rec["Expected_RenoCost"] = 26000; 

	## Renovate to Modern Kitchen
	elif feature_of_interest == 'Modern Kitchen':
		#R_KITCH_STYLE_L    0.012301 - luxury
		#R_KITCH_STYLE_M    0.022410 - Modern
		#R_KITCH_STYLE_S   -0.000000 - Semi-Modern
		#R_KITCH_STYLE_N   -0.027267 - No Remodeling
		#rec = rec[(rec["R_KITCH_STYLE_M"]!=1) & (rec["R_KITCH_STYLE_L"])!=1]
		BuildStylFeat = ['R_KITCH_STYLE_L', 'R_KITCH_STYLE_M','R_KITCH_STYLE_N', 'R_KITCH_STYLE_S']
		rec[BuildStylFeat] = 0;
		rec["R_KITCH_STYLE_M"] = 1;
		rec["Expected_RenoCost"] = 16000; 

	rec["MVReno19"] = np.exp(model.predict(rec[my_feat])); 
	rec["key"] = 2020;
	rec["MVReno20"] = np.exp(model.predict(rec[my_feat])); 
	rec["Prof19"] = rec["MVReno19"] - rec["MARKET_VALUE"] - rec["Expected_RenoCost"]
	rec["Prof20"] = rec["MVReno20"] - rec["MARKET_VALUE"] - rec["Expected_RenoCost"]
	#rec = rec.sample(n=60).sort_values(by="Prof19",ascending=False)
	#rec = rec[rec["Prof19"]<1e6]

	rec["MARKET_VALUE"] = rec["MARKET_VALUE"].astype(int);
	rec["MVReno19"] = rec["MVReno19"].astype(int);
	rec["MVReno20"] = rec["MVReno20"].astype(int);
	rec["Prof19"] = rec["Prof19"].astype(int);
	rec["Prof20"] = rec["Prof20"].astype(int);
	rec["Expected_RenoCost"] = rec["Expected_RenoCost"].astype(int);

	#print(len(dataf), len(rec))
	return rec["MVReno19"], rec["MVReno20"], rec["Prof19"], rec["Prof20"], rec["Expected_RenoCost"]



## -- Appreciation rate
def Appreciation(dataf,model,my_feat): 
	rec = dataf.copy(deep=True);
	rec["key"] = 2020;
	rec["Appreciation_2020"] = np.exp(model.predict(rec[my_feat])); 
	return rec["Appreciation_2020"]

## -- change integer to human readable 
def millify(n):
	millnames = ['',' Thousand',' Million',' Billion',' Trillion']
	n = float(n)
	millidx = max(0,min(len(millnames)-1,
		int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
	return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

## -- add zero for zipcode in Boston
def addressfix(val):
	val = val.split(".")[0]
	val = "0"+val 
	return val;

## -- starting code -- 
if __name__ == '__main__':
	sys.argv ## get the input argument
	total = len(sys.argv)
	cmdargs = sys.argv	
	main(total, cmdargs)










