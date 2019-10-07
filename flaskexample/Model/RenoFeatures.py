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

## Thank you stack overflow for millify function.
## https://stackoverflow.com/questions/3154460/python-human-readable-large-numbers
## user: Janus
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

def ModelIt2(feature_of_interest,query):

	## -- read in the data for relavent upgrades --
	inputfname="flaskexample/Model/AllReno.csv"
	recommend = pd.read_csv(inputfname)

	# -- convert prices to integer, if pandas read as object variable --
	recommend["MARKET_VALUE"] = recommend["MARKET_VALUE"].astype(int);

	#	for z in pd.unique(recommend["ZIPCODE"]):
	#		temper = recommend[recommend["ZIPCODE"]==z]
	#		print(z, len(temper))

	# -- adjust data to user defined query -- 
	recommend = recommend[recommend["ZIPCODE"]==int(query["zipcode"])]; 
	recommend = recommend[recommend["MARKET_VALUE"]<int(query["HomeValue_max"])]
	recommend = recommend[recommend["MARKET_VALUE"]>int(query["HomeValue_min"])]

	query["size"] = len(recommend);
	query["isZero"] = "please try again with different features" if len(recommend)==0 else " "; 
	return(recommend);


def ModelIt(feature_of_interest,query):

	## -- read in the data for relavent upgrades --
	inputfname = "0"
	if feature_of_interest == 'Upgrade to Excellent Interior Condition':
		inputfname = "flaskexample/Model/Int_Cond.csv"; 
	elif feature_of_interest == 'Upgrade to Good Exterior Condition':
		inputfname = "flaskexample/Model/Ext_Cond.csv";
	elif feature_of_interest == 'Upgrade to Elaborate Interior Finish':
		inputfname = "flaskexample/Model/Int_Fin.csv";
	elif feature_of_interest == 'Upgrade to Brick Exterior Finish':
		inputfname = "flaskexample/Model/Ext_Fin.csv";
	elif feature_of_interest == 'Add a Fireplace (why not!)':
		inputfname = "flaskexample/Model/Fire_Place.csv";
	elif feature_of_interest == 'Upgrade roof to Mansard type':
		inputfname = "flaskexample/Model/Roof_Upgrade.csv";
	elif feature_of_interest == 'Upgrade Kitchen to Luxury type':
		inputfname = "flaskexample/Model/L_Kitchen.csv";
	elif feature_of_interest == 'Upgrade Kitchen to Modern type':
		inputfname = "flaskexample/Model/M_Kitchen.csv";

	recommend = pd.read_csv(inputfname)


	# -- convert prices to integer, if pandas read as object variable --
	recommend["MARKET_VALUE"] = recommend["MARKET_VALUE"].astype(int);
	recommend["MVReno19"] = recommend["MVReno19"].astype(int);
	recommend["MVReno20"] = recommend["MVReno20"].astype(int);
	recommend["Prof19"] = recommend["Prof19"].astype(int);
	recommend["Prof20"] = recommend["Prof20"].astype(int);

	# -- calculated profit based on investment and user given renovation cost (not estimated renovation cost) --
	recommend["Prof19"] = recommend["Prof19"] - int(query["reno_cost"])
	recommend["Prof20"] = recommend["Prof20"] - int(query["reno_cost"])

	# -- adjust data to user defined query -- 
	recommend = recommend[recommend["ZIPCODE"]==int(query["zipcode"])]; 
	recommend = recommend[recommend["MARKET_VALUE"]<int(query["HomeValue_max"])]
	recommend = recommend[recommend["MARKET_VALUE"]>int(query["HomeValue_min"])]

	# -- show only profitable properties! -- 
	recommend = recommend[recommend["Prof19"]>0]; 
	recommend = recommend[recommend["Prof20"]>0]; 
	recommend = recommend.sort_values(by="Prof19",ascending=False)
	#recommend = recommend[recommend["Prof19"]<1e6]

	'''
	# Make a horizontal bar plot. 
	# blue  - current quik-flip estimate
	# green - renovation cost given by the user
	# red   - profit accessed using user given renovation cost. 
	scale=0.000001
	fig = plt.figure(figsize=(6, 10)); 
	ylabels = ["current value", "total investment", "2020 post-reno value"]
	counter=0; 
	for index, row in recommend.iterrows(): 
		val1 = row["MVReno20"]
		val2 = row["MARKET_VALUE"] + int(query["reno_cost"])
		val3 = row["MARKET_VALUE"]
		y_pos = [counter,counter,counter]; 
		values = np.asarray([val1,val2,val3])*scale
		plt.barh(y_pos, values, align="center",color=["red","green","blue"],height=0.9); 
		counter = counter+1; 

	yticks=[]
	if(len(recommend)<5): 
		yticks = np.arange(0,len(recommend),1)
	elif((len(recommend)>5) & (len(recommend)<10)): 
		yticks = np.arange(0,len(recommend),2)
	else:
		yticks = np.arange(0,len(recommend),5)


	plt.yticks(yticks,size=24) 
	xticks=np.arange(0,10,0.1)
	plt.xticks(xticks,size=24,rotation=90)
	plt.xlabel("$ amount (millions)",size=32)
	plt.ylabel("House index",size=32)

	plt.xlim(left = (recommend["MARKET_VALUE"].min()-110000)*scale, right=(recommend["MVReno20"].max()+110000)*scale)
	plt.ylim(top=-0.5,bottom=(len(recommend)+0.5))

	plt.savefig("flaskexample/Model/out.png",dpi=100,bbox_inches='tight')
	'''


	# -- Convert Variables to millions or thousands ---
	#recommend["FULLADD"] = recommend["FULLADD"].map(addressfix); 
	recommend["MARKET_VALUE"] = recommend["MARKET_VALUE"].map(millify); 
	recommend["MVReno19"] = recommend["MVReno19"].map(millify); 
	recommend["MVReno20"] = recommend["MVReno20"].map(millify); 
	recommend["Prof19"] = recommend["Prof19"].map(millify); 
	recommend["Prof20"] = recommend["Prof20"].map(millify); 
	recommend["Expected_RenoCost"] = recommend["Expected_RenoCost"] - int(query["reno_cost"]); #.map(millify); 
	#recommend["Expected_RenoCost"] = recommend["Expected_RenoCost"].map(millify); 
	query["size"] = len(recommend);
	query["isZero"] = "please try again with different features" if len(recommend)==0 else " "; 

	return(recommend);












