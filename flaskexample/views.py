""" This file largely follows the steps outlined in the Insight Flask tutorial, except data is stored in a
flat csv (./assets/births2012_downsampled.csv) vs. a postgres database. If you have a large database, or
want to build experience working with SQL databases, you should refer to the Flask tutorial for instructions on how to
query a SQL database from here instead.

May 2019, Donald Lee-Brown
"""

import sys,os,math,string,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import render_template
from flaskexample import app
#from flaskexample.a_model import ModelIt
from flask import request
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults
zillow_data = ZillowWrapper("ZILLOW_KEY")
from flaskexample.Model import RenoFeatures

@app.route('/')
def birthmodel_input():
	return render_template("index.html")


## Thank you stack overflow for millify function.
## https://stackoverflow.com/questions/3154460/python-human-readable-large-numbers
## user: Janus
def millify(n):
	millnames = ['',' Thousand',' Million',' Billion',' Trillion']
	n = float(n)
	millidx = max(0,min(len(millnames)-1,
		int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
	return '{:.3f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


@app.route('/reno_features')
def reno_features():
	qZipCode = request.args.get('zipcode')
	qReno_Feature = "0" #request.args.get('reno_feature')
	#qRenoCost = request.args.get('RenoCost')
	qHomeValue = request.args.get('HomeValue')


	qHomeValue_min = qHomeValue.split(" - ")[0].split("$")[1]
	qHomeValue_max = qHomeValue.split(" - ")[1].split("$")[1]
	#qRenoCost = qRenoCost.split("$")[1]
	print(" ================================================== ")
	print("zipcode = ", qZipCode)
	#print("reno_feature = ", qReno_Feature)
	print("HomeValue (min,max) ", qHomeValue_min, qHomeValue_max)
	#print("RenoValue = ", qRenoCost)
	print(" ================================================== ")

	query_parm = {}
	query_parm["zipcode"] = qZipCode; 
	query_parm["reno_feature"] = "0" # qReno_Feature;
	query_parm["HomeValue_min"] = qHomeValue_min;
	query_parm["HomeValue_max"] = qHomeValue_max;
	query_parm["reno_cost"] = "0" #qRenoCost; 

	recommendations = RenoFeatures.ModelIt2(qReno_Feature,query_parm)
	recommendations = recommendations.reset_index(drop=True); 

	recommendations["Appreciation_2020"] = recommendations["Appreciation_2020"].map(millify); 
	recommendations["MARKET_VALUE"] = recommendations["MARKET_VALUE"].map(millify); 
	recommendations["Prof20_INT_COND"] = recommendations["Prof20_INT_COND"].map(millify); 
	recommendations["Prof20_EXT_COND"] = recommendations["Prof20_EXT_COND"].map(millify); 
	recommendations["Prof20_INT_FIN"] = recommendations["Prof20_INT_FIN"].map(millify); 
	recommendations["Prof20_EXT_FIN"] = recommendations["Prof20_EXT_FIN"].map(millify); 
	recommendations["Prof20_FRPL"] = recommendations["Prof20_FRPL"].map(millify); 
	recommendations["Prof20_ROOF_M"] = recommendations["Prof20_ROOF_M"].map(millify); 
	recommendations["Prof20_KITCHEN_L"] = recommendations["Prof20_KITCHEN_L"].map(millify); 
	recommendations["Prof20_KITCHEN_M"] = recommendations["Prof20_KITCHEN_M"].map(millify);

	recommendations["MVReno20_INT_COND"] = recommendations["MVReno20_INT_COND"].map(millify); 
	recommendations["MVReno20_EXT_COND"] = recommendations["MVReno20_EXT_COND"].map(millify); 
	recommendations["MVReno20_INT_FIN"] = recommendations["MVReno20_INT_FIN"].map(millify); 
	recommendations["MVReno20_EXT_FIN"] = recommendations["MVReno20_EXT_FIN"].map(millify); 
	recommendations["MVReno20_FRPL"] = recommendations["MVReno20_FRPL"].map(millify); 
	recommendations["MVReno20_ROOF_M"] = recommendations["MVReno20_ROOF_M"].map(millify); 
	recommendations["MVReno20_KITCHEN_L"] = recommendations["MVReno20_KITCHEN_L"].map(millify); 
	recommendations["MVReno20_KITCHEN_M"] = recommendations["MVReno20_KITCHEN_M"].map(millify);

	recommendations["1yr_Increase"] = recommendations["1yr_Increase"].map(millify); 

	recommendations = recommendations.to_dict('index')
	dic = {}
	#	for x in recommendations[0].items(): 
	#		print(x)

	return render_template(
		'renovations_query.html',
		tables=recommendations, 
		query = query_parm
	)








# here's the homepage
#@app.route('/')
#def homepage():
#    return render_template("bootstrap_template.html")

## example page for linking things
#@app.route('/example_linked')
#def linked_example():
#    return render_template("example_linked.html")

##here's a page that simply displays the births data
#@app.route('/example_dbtable')
#def birth_table_page():
#    births = []
#    # let's read in the first 10 rows of births data - note that the directory is relative to run.py
#    dbname = './flaskexample/static/data/births2012_downsampled.csv'
#    births_db = pd.read_csv(dbname).head(10)
#    # when passing to html it's easiest to store values as dictionaries
#    for i in range(0, births_db.shape[0]):
#        births.append(dict(index=births_db.index[i], attendant=births_db.iloc[i]['attendant'],
#                           birth_month=births_db.iloc[i]['birth_month']))
#    # note that we pass births as a variable to the html page example_dbtable
#    return render_template('/example_dbtable.html', births=births)

## now let's do something fancier - take an input, run it through a model, and display the output on a separate page



#@app.route('/model_output')
#def birthmodel_output():
#   # pull 'birth_month' from input field and store it
#   patient = request.args.get('birth_month')

#   # read in our csv file
#   dbname = './flaskexample/static/data/births2012_downsampled.csv'
#   births_db = pd.read_csv(dbname)

#   # let's only select cesarean births with the specified birth month
#   births_db = births_db[births_db['delivery_method'] == 'Cesarean']
#   births_db = births_db[births_db['birth_month'] == patient]

#   # we really only need the attendant and birth month for this one
#   births_db = births_db[['attendant', 'birth_month']]

#   # just select the Cesareans  from the birth dtabase for the month that the user inputs
#   births = []
#   for i in range(0, births_db.shape[0]):
#      births.append(dict(index=births_db.index[i], attendant=births_db.iloc[i]['attendant'],
#                        birth_month=births_db.iloc[i]['birth_month']))
#   the_result = ModelIt(patient, births)
#   return render_template("model_output.html", births=births, the_result=the_result)
