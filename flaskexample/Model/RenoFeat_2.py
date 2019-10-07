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
	recommend["ROI_INT_COND"] = round(recommend["Prof20_INT_COND"]/(recommend["Expected_RenoCost_INT_COND"]+recommend["MARKET_VALUE"])*100,2); 
	
	feature_of_interest="Exterior Condition"
	recommend["MVReno19_EXT_COND"], recommend["MVReno20_EXT_COND"], recommend["Prof19_EXT_COND"], recommend["Prof20_EXT_COND"], recommend["Expected_RenoCost_EXT_COND"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_EXT_COND"] = round(recommend["Prof20_EXT_COND"]/(recommend["Expected_RenoCost_EXT_COND"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Interior Finish"
	recommend["MVReno19_INT_FIN"], recommend["MVReno20_INT_FIN"], recommend["Prof19_INT_FIN"], recommend["Prof20_INT_FIN"], recommend["Expected_RenoCost_INT_FIN"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_INT_FIN"] = round(recommend["Prof20_INT_FIN"]/(recommend["Expected_RenoCost_INT_FIN"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Exterior Finish: Brick"
	recommend["MVReno19_EXT_FIN_B"], recommend["MVReno20_EXT_FIN_B"], recommend["Prof19_EXT_FIN_B"], recommend["Prof20_EXT_FIN_B"], recommend["Expected_RenoCost_EXT_FIN_B"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_EXT_FIN_B"] = round(recommend["Prof20_EXT_FIN_B"]/(recommend["Expected_RenoCost_EXT_FIN_B"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Exterior Finish: Cement"
	recommend["MVReno19_EXT_FIN_C"], recommend["MVReno20_EXT_FIN_C"], recommend["Prof19_EXT_FIN_C"], recommend["Prof20_EXT_FIN_C"], recommend["Expected_RenoCost_EXT_FIN_C"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_EXT_FIN_C"] = round(recommend["Prof20_EXT_FIN_C"]/(recommend["Expected_RenoCost_EXT_FIN_C"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Fireplace"
	recommend["MVReno19_FRPL"], recommend["MVReno20_FRPL"], recommend["Prof19_FRPL"], recommend["Prof20_FRPL"], recommend["Expected_RenoCost_FRPL"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_FRPL"] = round(recommend["Prof20_FRPL"]/(recommend["Expected_RenoCost_FRPL"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Mansard Roof"
	recommend["MVReno19_ROOF_M"], recommend["MVReno20_ROOF_M"], recommend["Prof19_ROOF_M"], recommend["Prof20_ROOF_M"], recommend["Expected_RenoCost_ROOF_M"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_ROOF_M"] = round(recommend["Prof20_ROOF_M"]/(recommend["Expected_RenoCost_ROOF_M"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Luxury Kitchen"
	recommend["MVReno19_KITCHEN_L"], recommend["MVReno20_KITCHEN_L"], recommend["Prof19_KITCHEN_L"], recommend["Prof20_KITCHEN_L"], recommend["Expected_RenoCost_KITCHEN_L"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_KITCHEN_L"] = round(recommend["Prof20_KITCHEN_L"]/(recommend["Expected_RenoCost_KITCHEN_L"]+recommend["MARKET_VALUE"])*100,2); 

	feature_of_interest="Modern Kitchen"
	recommend["MVReno19_KITCHEN_M"], recommend["MVReno20_KITCHEN_M"], recommend["Prof19_KITCHEN_M"], recommend["Prof20_KITCHEN_M"], recommend["Expected_RenoCost_KITCHEN_M"] = Renovate(recommend,ElasticNetCVModel2,feature_of_interest,my_feat); 
	recommend["ROI_KITCHEN_M"] = round(recommend["Prof20_KITCHEN_M"]/(recommend["Expected_RenoCost_KITCHEN_M"]+recommend["MARKET_VALUE"])*100,2); 

	recommend['Max_Prof'] = recommend.apply(lambda row: np.max([row['Prof20_INT_COND'],row['Prof20_EXT_COND'],row['Prof20_INT_FIN'],row['Prof20_EXT_FIN_B'],row['Prof20_EXT_FIN_C'],row['Prof20_FRPL'],row['Prof20_ROOF_M'],row['Prof20_KITCHEN_L'],row['Prof20_KITCHEN_M']]), axis=1)

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
	elif feature_of_interest == 'Exterior Finish: Brick':
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
		rec["Expected_RenoCost"] = 7500*rec["NUM_FLOORS"];

	## -- Renovate Exterior Finish
	elif feature_of_interest == 'Exterior Finish: Cement':
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
		rec["R_EXT_FIN_C"] = 1;
		rec["Expected_RenoCost"] = 9000*rec["NUM_FLOORS"];

#	## -- Renovate Exterior Finish
#	elif feature_of_interest == 'Exterior Finish':
#		#R_EXT_FIN_A       -0.083155
#		#R_EXT_FIN_B        0.000000
#		#R_EXT_FIN_C        0.041162
#		#R_EXT_FIN_F       -0.000000
#		#R_EXT_FIN_G        0.171578
#		#R_EXT_FIN_M       -0.021715
#		#R_EXT_FIN_O       -0.000000
#		#R_EXT_FIN_P       -0.108465
#		#R_EXT_FIN_S        0.015849
#		#R_EXT_FIN_U       -0.028644
#		#R_EXT_FIN_V        0.023906
#		#R_EXT_FIN_W        0.016915
#		#rec = rec[rec["R_EXT_FIN_B"]!=1]
#		BuildStylFeat = ['R_EXT_FIN_A', 'R_EXT_FIN_B','R_EXT_FIN_C', 'R_EXT_FIN_F', 'R_EXT_FIN_G', 'R_EXT_FIN_M','R_EXT_FIN_O', 'R_EXT_FIN_P', 'R_EXT_FIN_S', 'R_EXT_FIN_U','R_EXT_FIN_V', 'R_EXT_FIN_W']
#		rec[BuildStylFeat] = 0;
#		rec["R_EXT_FIN_B"] = 1;
#		rec["Expected_RenoCost"] = 5000*rec["NUM_FLOORS"];

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
















#	LU_R1              0.026575
#	LU_R2             -0.000000
#	LU_R3             -0.028450
#	R_BLDG_STYL_BW    -0.053931
#	R_BLDG_STYL_CL    -0.000000
#	R_BLDG_STYL_CN     0.062834
#	R_BLDG_STYL_CP     0.019837
#	R_BLDG_STYL_CV    -0.029216
#	R_BLDG_STYL_DK    -0.030753
#	R_BLDG_STYL_DX     0.000000
#	R_BLDG_STYL_OT     0.160699
#	R_BLDG_STYL_RE     0.036670
#	R_BLDG_STYL_RM     0.019177
#	R_BLDG_STYL_RN     0.034781
#	R_BLDG_STYL_RR    -0.015725
#	R_BLDG_STYL_SD     0.011939
#	R_BLDG_STYL_SL     0.044960
#	R_BLDG_STYL_TD     0.122100
#	R_BLDG_STYL_TF    -0.034497
#	R_BLDG_STYL_TL     0.003140
#	R_BLDG_STYL_VT    -0.000000
#	R_ROOF_TYP_F      -0.003195
#	R_ROOF_TYP_G       0.000000
#	R_ROOF_TYP_H       0.011795
#	R_ROOF_TYP_L      -0.008242
#	R_ROOF_TYP_M       0.043858
#	R_ROOF_TYP_S      -0.029693
#	R_EXT_FIN_A       -0.082516
#	R_EXT_FIN_B        0.000000
#	R_EXT_FIN_C        0.046371
#	R_EXT_FIN_F        0.000643
#	R_EXT_FIN_G        0.189047
#	R_EXT_FIN_M       -0.020841
#	R_EXT_FIN_O       -0.000946
#	R_EXT_FIN_P       -0.108641
#	R_EXT_FIN_S        0.019253
#	R_EXT_FIN_U       -0.031405
#	R_EXT_FIN_V        0.031115
#	R_EXT_FIN_W        0.018454
#	R_BTH_STYLE_L     -0.000000
#	R_BTH_STYLE_M      0.008643
#	R_BTH_STYLE_N     -0.009705
#	R_BTH_STYLE_S     -0.000000
#	R_KITCH_STYLE_L    0.013057
#	R_KITCH_STYLE_M    0.023166
#	R_KITCH_STYLE_N   -0.027441
#	R_KITCH_STYLE_S   -0.000000
#	R_HEAT_TYP_E      -0.068241
#	R_HEAT_TYP_F      -0.015736
#	R_HEAT_TYP_N       0.008574
#	R_HEAT_TYP_O      -0.090271
#	R_HEAT_TYP_P       0.028861
#	R_HEAT_TYP_S       0.010774
#	R_HEAT_TYP_W       0.000000
#	R_AC_C             0.000000
#	R_AC_D            -0.091346
#	R_AC_N            -0.058683
#	R_AC_Y             0.010727
#	R_EXT_CND_A       -0.000000
#	R_EXT_CND_E       -0.047321
#	R_EXT_CND_F       -0.010090
#	R_EXT_CND_G        0.013060
#	R_EXT_CND_P       -0.156119
#	R_OVRALL_CND_A     0.003151
#	R_OVRALL_CND_E    -0.050977
#	R_OVRALL_CND_F    -0.018061
#	R_OVRALL_CND_G    -0.000000
#	R_OVRALL_CND_P     0.029905
#	R_INT_CND_A       -0.038888
#	R_INT_CND_E        0.081361
#	R_INT_CND_F       -0.064918
#	R_INT_CND_G        0.006824
#	R_INT_CND_P       -0.056216
#	R_INT_FIN_E        0.093781
#	R_INT_FIN_N       -0.000000
#	R_INT_FIN_S       -0.100145
#	R_VIEW_A          -0.000000
#	R_VIEW_E           0.226984
#	R_VIEW_F          -0.040035
#	R_VIEW_G           0.100377
#	R_VIEW_P           0.016614
#	PTYPE_101.0        0.000578
#	PTYPE_104.0       -0.000000
#	PTYPE_105.0       -0.002954
#	OWN_OCC            0.006883
#	LAND_SF            0.000014
#	YR_BUILT          -0.000323
#	YR_REMOD           0.000013
#	GROSS_AREA         0.151551
#	LIVING_AREA        0.300397
#	NUM_FLOORS         0.032868
#	R_TOTAL_RMS        0.002771
#	R_BDRMS           -0.004786
#	R_FULL_BTH         0.045067
#	R_HALF_BTH         0.030981
#	R_KITCH            0.007677
#	R_FPLACE           0.015375
#	R_TOTAL_BTH        0.007494
#	DIS0              -0.008269
#	DIS1              -0.032259
#	DIS2               0.031144
#	DIS3              -0.000000
#	DIS4              -0.099429
#	DIS5               0.085343
#	ZIP_MV             0.773274
#	key                0.005713
#	dtype: float64
#	train R^2 =  0.9282881818016006
#	test R^2 =  0.9273707866836075
#	L1_ratio =  1.0
#	ZIP_MV             0.773274
#	LIVING_AREA        0.300397
#	R_VIEW_E           0.226984
#	R_EXT_FIN_G        0.189047
#	R_BLDG_STYL_OT     0.160699
#	GROSS_AREA         0.151551
#	R_BLDG_STYL_TD     0.122100
#	R_VIEW_G           0.100377
#	R_INT_FIN_E        0.093781
#	DIS5               0.085343
#	R_INT_CND_E        0.081361
#	R_BLDG_STYL_CN     0.062834
#	R_EXT_FIN_C        0.046371
#	R_FULL_BTH         0.045067
#	R_BLDG_STYL_SL     0.044960
#	R_ROOF_TYP_M       0.043858
#	R_BLDG_STYL_RE     0.036670
#	R_BLDG_STYL_RN     0.034781
#	NUM_FLOORS         0.032868
#	DIS2               0.031144
#	R_EXT_FIN_V        0.031115
#	R_HALF_BTH         0.030981
#	R_OVRALL_CND_P     0.029905
#	R_HEAT_TYP_P       0.028861
#	LU_R1              0.026575
#	R_KITCH_STYLE_M    0.023166
#	R_BLDG_STYL_CP     0.019837
#	R_EXT_FIN_S        0.019253
#	R_BLDG_STYL_RM     0.019177
#	R_EXT_FIN_W        0.018454
#	R_VIEW_P           0.016614
#	R_FPLACE           0.015375
#	R_EXT_CND_G        0.013060
#	R_KITCH_STYLE_L    0.013057
#	R_BLDG_STYL_SD     0.011939
#	R_ROOF_TYP_H       0.011795
#	R_HEAT_TYP_S       0.010774
#	R_AC_Y             0.010727
#	R_BTH_STYLE_M      0.008643
#	R_HEAT_TYP_N       0.008574
#	R_KITCH            0.007677
#	R_TOTAL_BTH        0.007494
#	OWN_OCC            0.006883
#	R_INT_CND_G        0.006824
#	key                0.005713
#	R_OVRALL_CND_A     0.003151
#	R_BLDG_STYL_TL     0.003140
#	R_TOTAL_RMS        0.002771
#	R_EXT_FIN_F        0.000643
#	PTYPE_101.0        0.000578
#	LAND_SF            0.000014
#	YR_REMOD           0.000013
#	R_EXT_CND_A       -0.000000
#	R_VIEW_A          -0.000000
#	R_OVRALL_CND_G    -0.000000
#	R_BLDG_STYL_VT    -0.000000
#	R_ROOF_TYP_G       0.000000
#	R_EXT_FIN_B        0.000000
#	LU_R2             -0.000000
#	PTYPE_104.0       -0.000000
#	DIS3              -0.000000
#	R_BTH_STYLE_L     -0.000000
#	R_BTH_STYLE_S     -0.000000
#	R_BLDG_STYL_DX     0.000000
#	R_BLDG_STYL_CL    -0.000000
#	R_AC_C             0.000000
#	R_HEAT_TYP_W       0.000000
#	R_KITCH_STYLE_S   -0.000000
#	R_INT_FIN_N       -0.000000
#	YR_BUILT          -0.000323
#	R_EXT_FIN_O       -0.000946
#	PTYPE_105.0       -0.002954
#	R_ROOF_TYP_F      -0.003195
#	R_BDRMS           -0.004786
#	R_ROOF_TYP_L      -0.008242
#	DIS0              -0.008269
#	R_BTH_STYLE_N     -0.009705
#	R_EXT_CND_F       -0.010090
#	R_BLDG_STYL_RR    -0.015725
#	R_HEAT_TYP_F      -0.015736
#	R_OVRALL_CND_F    -0.018061
#	R_EXT_FIN_M       -0.020841
#	R_KITCH_STYLE_N   -0.027441
#	LU_R3             -0.028450
#	R_BLDG_STYL_CV    -0.029216
#	R_ROOF_TYP_S      -0.029693
#	R_BLDG_STYL_DK    -0.030753
#	R_EXT_FIN_U       -0.031405
#	DIS1              -0.032259
#	R_BLDG_STYL_TF    -0.034497
#	R_INT_CND_A       -0.038888
#	R_VIEW_F          -0.040035
#	R_EXT_CND_E       -0.047321
#	R_OVRALL_CND_E    -0.050977
#	R_BLDG_STYL_BW    -0.053931
#	R_INT_CND_P       -0.056216
#	R_AC_N            -0.058683
#	R_INT_CND_F       -0.064918
#	R_HEAT_TYP_E      -0.068241
#	R_EXT_FIN_A       -0.082516
#	R_HEAT_TYP_O      -0.090271
#	R_AC_D            -0.091346
#	DIS4              -0.099429
#	R_INT_FIN_S       -0.100145
#	R_EXT_FIN_P       -0.108641
#	R_EXT_CND_P       -0.156119





