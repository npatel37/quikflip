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

	for z in pd.unique(recommend["ZIPCODE"]):
		temper = recommend[recommend["ZIPCODE"]==z]
		print(z, len(temper))

	# -- adjust data to user defined query -- 
	recommend = recommend[recommend["ZIPCODE"]==int(query["zipcode"])]; 
	recommend = recommend[recommend["MARKET_VALUE"]<int(query["HomeValue_max"])]
	recommend = recommend[recommend["MARKET_VALUE"]>int(query["HomeValue_min"])]

	query["size"] = len(recommend);
	query["isZero"] = "please try again with different features" if len(recommend)==0 else " "; 
	return(recommend);
