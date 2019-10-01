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

#R_KITCH_STYLE_L    0.012301 - luxury
#R_KITCH_STYLE_M    0.022410 - Modern
#R_KITCH_STYLE_S   -0.000000 - Semi-Modern
#R_KITCH_STYLE_N   -0.027267 - No Remodeling

def millify(n):
	millnames = ['',' Thousand',' Million',' Billion',' Trillion']
	n = float(n)
	millidx = max(0,min(len(millnames)-1,
		int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

	return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def addressfix(val):
	val = val.split(".")[0]
	val = "0"+val 
	return val;

##### ----------------------
def main(total, cmdargs):
	if total != 1:
		print (" ".join(str(x) for x in cmdargs))
		raise ValueError('I did not ask for arguments')
	os.system("pwd")
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

	filename = 'finalized_model.sav'
	ElasticNetCVModel2 = pickle.load(open(filename, 'rb'))

	df3["Residual"] =  -np.exp(ElasticNetCVModel2.predict(df3[my_feat])) + np.exp(df3["MARKET_VALUE"])

	## prediction - marketvalue ---> (negative is under-valued!)
	predic_y = np.exp(ElasticNetCVModel2.predict(df3[my_feat]))
	df3["Residual"] =  predic_y - np.exp(df3["MARKET_VALUE"])

	## -- find recommendations ---
	recommend = df3[(df3["key"]==2019)]
	recommend = recommend[(recommend["Residual"]<50000) & (recommend["Residual"]>-12000)]
	recommend = recommend[(recommend["R_KITCH_STYLE_M"]!=1) & (recommend["R_KITCH_STYLE_L"])!=1]

	BuildStylFeat = ['R_KITCH_STYLE_L', 'R_KITCH_STYLE_M','R_KITCH_STYLE_N', 'R_KITCH_STYLE_S']
	recommend[BuildStylFeat] = 0;
	recommend["R_KITCH_STYLE_M"] = 1;
	recommend["MARKET_VALUE"] = np.exp(recommend["MARKET_VALUE"]); 
	recommend["MVReno19"] = np.exp(ElasticNetCVModel2.predict(recommend[my_feat])); 
	recommend["key"] = 2020;
	recommend["MVReno20"] = np.exp(ElasticNetCVModel2.predict(recommend[my_feat])); 
	recommend["Prof19"] = recommend["MVReno19"] - recommend["MARKET_VALUE"]; 
	recommend["Prof20"] = recommend["MVReno20"] - recommend["MARKET_VALUE"]; 

	#recommend = recommend.sample(n=60).sort_values(by="Prof19",ascending=False)
	#recommend = recommend[recommend["Prof19"]<1e6]

	recommend["MARKET_VALUE"] = recommend["MARKET_VALUE"].astype(int);
	recommend["MVReno19"] = recommend["MVReno19"].astype(int);
	recommend["MVReno20"] = recommend["MVReno20"].astype(int);
	recommend["Prof19"] = recommend["Prof19"].astype(int);
	recommend["Prof20"] = recommend["Prof20"].astype(int);
	recommend["Expected_RenoCost"] = 20000; 
	recommend["Expected_RenoCost"] = recommend["Expected_RenoCost"].astype(int);

	#recommend["FULLADD"] = recommend["FULLADD"].map(addressfix); 
	#recommend["MARKET_VALUE"] = recommend["MARKET_VALUE"].map(millify); 
	#recommend["MVReno19"] = recommend["MVReno19"].map(millify); 
	#recommend["MVReno20"] = recommend["MVReno20"].map(millify); 
	#recommend["Prof19"] = recommend["Prof19"].map(millify); 
	#recommend["Prof20"] = recommend["Prof20"].map(millify); 

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
	recommend.to_csv("M_Kitchen.csv",index=False); 
	#return(recommend);




	
if __name__ == '__main__':
	sys.argv ## get the input argument
	total = len(sys.argv)
	cmdargs = sys.argv	
	main(total, cmdargs)










